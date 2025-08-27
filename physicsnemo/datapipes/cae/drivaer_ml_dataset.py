# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
import torch
import zarr

try:
    import tensorstore as ts

    TENSORSTORE_AVAILABLE = True
except ImportError:
    TENSORSTORE_AVAILABLE = False

from physicsnemo.distributed import ShardTensor, ShardTensorSpec

# from physicsnemo.distributed.utils import compute_split_shapes

# For use on systems where cpu_affinity is not available:
psutil_process = psutil.Process()


class FakeProcess:
    """
    Enable a fake cpu affinity setting if it's not available
    """

    def cpu_affinity(self, cpus: list[int] | None) -> None:
        pass


if not hasattr(psutil_process, "cpu_affinity"):
    psutil_process = FakeProcess()

# Abstractions:
# - want to read npy/npz/.zarr/.stl/.vtp files
# - Need to share next level abstractions
# - Domain parallel dataloading is supported: output will be ShardTensor instead.
# - need to be able to configure preprocessing
# - CPU -> GPU transfer happens here, needs to be isolated in it's own stream
# - Output of dataloader should be torch.Tensor objects.


"""
This datapipe handles reading files from Zarr and piping into torch.Tensor objects.

It's expected that the files are organized as groups, with each .zarr
file representing one training example.  To improve IO performance, the files 
should be chunked for each array.  The reader takes a list of keys in the 
group to read, and will not read keys that are not specified.  The exception
is if _no_ keys are passed, in which case _all_ keys will be read.
"""


class BackendReader(ABC):
    """
    Abstract base class for backend readers.
    """

    def __init__(self, keys_to_read: list[str] | None) -> None:
        """
        Initialize the backend reader.
        """
        self.keys_to_read = keys_to_read

    @abstractmethod
    def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        pass

    @abstractmethod
    def read_file_sharded(
        self, filename: pathlib.Path, parallel_rank: int, parallel_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, ShardTensorSpec]]:
        """
        Read a file and return a dictionary of tensors.
        """
        pass


class NpyFileReader(BackendReader):
    """
    Reader for numpy files.
    """

    def __init__(self, keys_to_read: list[str] | None) -> None:
        super().__init__(keys_to_read)

    def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        data = np.load(filename, allow_pickle=True).item()

        missing_keys = set(self.keys_to_read) - set(data.keys())

        if len(missing_keys) > 0:
            raise ValueError(f"Keys {missing_keys} not found in file {filename}")

        data = {key: torch.from_numpy(data[key]) for key in self.keys_to_read}

        return data

    def read_file_sharded(
        self, filename: pathlib.Path, parallel_rank: int, parallel_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, ShardTensorSpec]]:
        pass


class ZarrFileReader(BackendReader):
    """
    Reader for zarr files.
    """

    def __init__(self, keys_to_read: list[str] | None) -> None:
        super().__init__(keys_to_read)

    def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        group = zarr.open_group(filename, mode="r")

        missing_keys = set(self.keys_to_read) - set(group.keys())

        if len(missing_keys) > 0:
            raise ValueError(f"Keys {missing_keys} not found in file {filename}")

        # This is a slower basic way to do this, to be improved:
        data = {key: torch.from_numpy(group[key][:]) for key in self.keys_to_read}

        return data

    def read_file_sharded(
        self, filename: pathlib.Path, parallel_rank: int, parallel_size: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, ShardTensorSpec]]:
        """
        Read a file and return a dictionary of tensors.
        """
        pass


if TENSORSTORE_AVAILABLE:

    class TensorStoreZarrReader(BackendReader):
        """
        Reader for tensorstore zarr files.
        """

        def __init__(self, keys_to_read: list[str] | None) -> None:
            super().__init__(keys_to_read)

            self.spec_template = {
                "driver": "zarr2",
                "kvstore": {
                    "driver": "file",
                    "path": None,
                },
            }

            self.context = ts.Context(
                {
                    "cache_pool": {"total_bytes_limit": 30_000_000},
                    "data_copy_concurrency": {"limit": 60},
                }
            )

        def read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
            """
            Read a file and return a dictionary of tensors.
            """
            read_futures = {}
            for key in self.keys_to_read:
                spec = self.spec_template.copy()
                spec["kvstore"]["path"] = str(filename) + "/" + str(key)

                read_futures[key] = ts.open(
                    spec, create=False, open=True, context=self.context
                )

            results = {
                key: np.array(read_futures[key].result()) for key in self.keys_to_read
            }

            data = {
                key: torch.as_tensor(results[key], dtype=torch.float32)
                for key in self.keys_to_read
            }

            return data

        def read_file_sharded(
            self, filename: pathlib.Path, parallel_rank: int, parallel_size: int
        ) -> tuple[dict[str, torch.Tensor], dict[str, ShardTensorSpec]]:
            """
            Read a file and return a dictionary of tensors.
            """
            pass
else:

    class TensorStoreZarrReader(BackendReader):
        """
        Null reader for tensorstore zarr files.
        """

        pass


class DrivaerMLDataset:
    """
    Dataset reader for DrivaerML and similar datasets.  In general, this
    dataset supports reading dictionary-like data, and returning a
    dictionary of torch.Tensor objects.

    When constructed, the user must pass a directory of data examples.
    The dataset will inspect the folder, identify all children, and decide:
    - If every file is a directory ending in .zarr, the zarr reader is used.
    - If every file is .npy, the .npy reader is used.
    - If every file is .npz, the .npz reader is used.
    - If every file is a directory without an extension, it's assumed to be .stl/.vtp/.vtu

    The user can optionally force one path with a parameter.

    The flow of this dataset is:
    - Load data from file, using a thread.
        - Each individual file reading tool may or may not have it's own threading
          or multi processing enabled.  That's up to it.  This just does async
          loading.
        - Data should come out of the readers in dict{str : torch.Tensor} format
    - The data is transferred from CPU to GPU in a separate stream.

    Users can call __getitem__(i), which will trigger the pipeline,
    or they can call `preload(i)`, which will start the pipeline for index `i`.
    Subsequent calls to `__getitem__(i)` should be faster since the IO is in
    progress or complete.

    Using the `__iter__` functionality will automatically enable preloading.

    """

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        keys_to_read: list[str] | None,
        output_device: torch.device,
        preload_depth: int = 2,
        device_mesh: torch.distributed.DeviceMesh | None = None,
        placements: dict[str, torch.distributed.tensor.Placement] | None = None,
        consumer_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if isinstance(data_dir, str):
            data_dir = pathlib.Path(data_dir)

        # Verify the data directory exists:
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")

        # Verify the data directory is a directory:
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Data directory {data_dir} is not a directory")

        self._keys_to_read = keys_to_read
        self.file_reader, self._filenames = self._infer_file_type_and_filenames(
            data_dir
        )

        # Check the file names; some can be read well in parallel, while others
        # are not parallelizable.

        self._length = len(self._filenames)

        self.output_device = output_device
        if output_device.type == "cuda":
            self._data_loader_stream = torch.cuda.Stream()
        else:
            self._data_loader_stream = None

        self.device_mesh = device_mesh
        self.placements = placements

        # This is thread storage for data preloading:
        self._preload_queue = {}
        self._transfer_events = {}
        self.preload_depth = preload_depth
        self.preload_executor = ThreadPoolExecutor(max_workers=preload_depth)

        if consumer_stream is None and self.output_device.type == "cuda":
            consumer_stream = torch.cuda.current_stream()

        self.consumer_stream = consumer_stream

    def _infer_file_type_and_filenames(
        self, data_dir: pathlib.Path
    ) -> tuple[str, list[str]]:
        """
        Infer the file type and filenames from the data directory.
        """

        # We validated the directory exists and is a directory already.

        # List the files:
        files = list(data_dir.iterdir())

        # Initialize the file reader object
        # Note that for some of these, they could be functions
        # But others benefit from having a state, so we use classes:

        if all(file.suffix == ".npy" for file in files):
            file_reader = NpyFileReader(self._keys_to_read)
            return file_reader, files
        elif all(file.suffix == ".zarr" and file.is_dir() for file in files):
            if TENSORSTORE_AVAILABLE:
                file_reader = TensorStoreZarrReader(self._keys_to_read)
            else:
                file_reader = ZarrFileReader(self._keys_to_read)
            return file_reader, files
        else:
            # TODO - support folders of stl, vtp, vtu.
            raise ValueError(f"Unsupported file type: {files}")

    def _move_to_gpu(
        self, data: dict[str, torch.Tensor], idx: int
    ) -> dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors and move to GPU if available.

        Args:
            data: Dictionary of key to torch tensor.

        Returns:
            Dictionary of key to torch tensor on GPU if available.
        """

        if self.output_device.type != "cuda":
            return data

        # result = StreamDict()
        result = {}

        with torch.cuda.stream(self._data_loader_stream):
            for key in data.keys():
                # Move to GPU if available
                result[key] = data[key].to(self.output_device, non_blocking=True)
                result[key].record_stream(self.consumer_stream)
            # Mark the consumer stream:
            transfer_event = torch.cuda.Event()
            transfer_event.record(self._data_loader_stream)
            # result.set_event("transfer", transfer_event)

        return result

    def _convert_to_shard_tensors(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, ShardTensor]:
        """Convert tensors to ShardTensor objects for distributed training.

        Args:
            tensors: Dictionary of key to torch tensor.

        Returns:
            Dictionary of key to torch tensor or ShardTensor.
        """

        if self.device_mesh is None:
            return tensors

        raise NotImplementedError("Converting to ShardTensor here not implemented yet.")

        # result = {}

        # for key, tensor in tensors.items():
        #     # Create a ShardTensor with whatever layout the data is actually in:
        #     st = ShardTensor.__new__(
        #         ShardTensor,
        #         local_tensor=tensor,
        #         spec=self.tensor_specs[key],
        #         requires_grad=False,  # By default, the data pipe output doesn't need a grad.
        #     )

        #     # Find out the desired placement:
        #     if tensor.numel() > 1:
        #         if isinstance(self.placements, dict):
        #             target_placement = self.placements[key]
        #         else:
        #             target_placement = self.placements
        #     else:
        #         target_placement = (Replicate(),)

        #     # Redistribute if necessary:
        #     # (Recall that this is one dimensional mesh only)
        #     if st._spec.placements[0] != target_placement[0]:
        #         st = st.redistribute(placements=target_placement)

        #     result[key] = st

        # return result

    def preload(self, idx: int) -> None:
        """
        Asynchronously preload the data for the given index (up to CPU, not GPU).
        Only one preload operation is supported at a time.

        Args:
            idx: Index of the sample to preload.
        """
        if idx in self._preload_queue:
            # Skip items that are already in the queue
            return

        def _preload_worker():
            data = self._read_file(self._filenames[idx])
            # Convert to torch tensors
            return self._move_to_gpu(data, idx)

        self._preload_queue[idx] = self.preload_executor.submit(_preload_worker)

    def get_preloaded(self, idx: int) -> dict[str, torch.Tensor] | None:
        """
        Retrieve the preloaded data (blocking if not ready).

        Returns:
            (idx, data) tuple where data is a dictionary of key to numpy array or torch tensor.

        Raises:
            RuntimeError: If no preload is in progress.
            Exception: If preload failed.
        """

        if idx not in self._preload_queue:
            return None

        result = self._preload_queue[
            idx
        ].result()  # This will block until the result is ready
        self._preload_queue.pop(idx)  # Clear the future after getting the result

        return result

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self._filenames):
            self.i = 0
            raise StopIteration

        if self.preload_depth > 0 and self.i + 1 < len(self._filenames):
            self.preload(self.i + 1)
        if self.preload_depth > 1 and self.i + 2 < len(self._filenames):
            self.preload(self.i + 2)

        data = self.__getitem__(self.i)

        self.i += 1

        return data

    def __len__(self):
        return len(self._filenames)

    def _read_file(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:
        """
        Read a file and return a dictionary of tensors.
        """
        return self.file_reader.read_file(filename)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | ShardTensor]:
        """
        Get a data sample.

        Flow is:
        - Read data, or get preloaded data if this idx is preloaded.
        - Move data to GPU, if needed.
            - Preloading data will move to GPU if it can.
        - If domain parallelism is enabled, convert to ShardTensors.
        - Return

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing tensors/ShardTensors for the requested data
        """

        if idx >= len(self._filenames):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self._filenames)}"
            )

        # Attempt to get preloaded data:
        data = self.get_preloaded(idx)
        if data is None:
            # Read data from zarr file
            data = self._read_file(self._filenames[idx])
            data = self._move_to_gpu(data, idx)

        # This blocks until the preprocessing has transferred to GPU
        if idx in self._transfer_events:
            self.consumer_stream.wait_event(self._transfer_events[idx])
            self._transfer_events.pop(idx)

        # Convert to ShardTensors if using domain parallelism
        if self.device_mesh is not None:
            data = self._convert_to_shard_tensors(data)

        return data


def compute_mean_std_min_max(
    dataset: DrivaerMLDataset, field_keys: list[str], max_samples: int = 20
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the mean, standard deviation, minimum, and maximum for a specified field
    across all samples in a dataset.

    Uses a numerically stable online algorithm for mean and variance.

    Args:
        dataset (DrivaerMLDataset): The dataset to process.
        field_key (str): The key for the field to normalize.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            mean, std, min, max tensors for the field.
    """
    N = {}
    mean = {}
    M2 = {}  # Sum of squares of differences from the current mean
    min_val = {}
    max_val = {}

    # Read the first data item to get the shapes:
    example_data = dataset[0]

    # Create placeholders for the accumulators:
    for key in field_keys:
        N[key] = torch.zeros(1, dtype=torch.int64, device=example_data[key].device)
        mean[key] = torch.zeros(
            example_data[key].shape[-1],
            device=example_data[key].device,
            dtype=torch.float64,
        )
        M2[key] = torch.zeros(
            example_data[key].shape[-1],
            device=example_data[key].device,
            dtype=torch.float64,
        )
        min_val[key] = torch.full(
            (example_data[key].shape[-1],),
            float("inf"),
            device=example_data[key].device,
        )
        max_val[key] = torch.full(
            (example_data[key].shape[-1],),
            float("-inf"),
            device=example_data[key].device,
        )

    global_start = time.perf_counter()
    start = time.perf_counter()
    for i, data in enumerate(dataset):
        if i >= max_samples:
            break

        for field_key in field_keys:
            field_data = data[field_key]

            # Compute batch statistics
            batch_mean = field_data.mean(axis=(0))
            batch_M2 = ((field_data - batch_mean) ** 2).sum(axis=(0))
            batch_n = field_data.shape[0]

            # Update min/max
            batch_min = field_data.amin(dim=(0))
            batch_max = field_data.amax(dim=(0))
            min_val[field_key] = torch.minimum(min_val[field_key], batch_min)
            max_val[field_key] = torch.maximum(max_val[field_key], batch_max)

            # Update running mean and M2 (Welford's algorithm)
            delta = batch_mean - mean[field_key]
            N[field_key] += batch_n  # batch_n should also be torch.int64
            mean[field_key] = mean[field_key] + delta * (batch_n / N[field_key])
            M2[field_key] = (
                M2[field_key]
                + batch_M2
                + delta**2 * (batch_n * N[field_key]) / N[field_key]
            )

        end = time.perf_counter()
        iteration_time = end - start
        print(f"on iteration {i} of {max_samples}, time: {iteration_time:.2f} seconds")
        start = time.perf_counter()

    global_end = time.perf_counter()
    global_time = global_end - global_start

    print(f"Total time: {global_time:.2f} seconds for {max_samples} samples")

    var = {}
    std = {}
    for field_key in field_keys:
        var[field_key] = M2[field_key] / (
            N[field_key].item() - 1
        )  # Convert N to Python int for division
        std[field_key] = torch.sqrt(var[field_key])

    return mean, std, min_val, max_val
