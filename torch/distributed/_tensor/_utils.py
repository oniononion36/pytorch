from typing import cast, List, Sequence, Tuple

import torch
import torch.distributed._tensor.api as dtensor
from torch._prims_common import ShapeType
from torch.distributed._tensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.device_mesh import DeviceMesh


# TODO: audit existing code base to see if we can safely remove this API.
def compute_local_shape(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[int, ...]:
    """
    Compute the shape of a local shard of the given DTensor on its current
    coordinate of the mesh.
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty shape
        return (0,)
    else:
        local_shape = list(global_shape)  # start with global shape
        ndim = len(global_shape)
        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert (
                    shard_dim < ndim
                ), f"Sharding dim {shard_dim} greater than tensor ndim {ndim}"
                local_shard_size, _ = placement._local_shard_size_on_dim(
                    local_shape[shard_dim], mesh_dim_size, my_coordinate[idx]
                )
                assert isinstance(local_shard_size, int)
                local_shape[shard_dim] = local_shard_size

        return tuple(local_shape)


def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.

    Example (2 host with 4GPUs each):
    # Below is a DeviceMesh with mesh_shape of (2, 4)
    mesh = DeviceMesh(device_type="cuda",
                        mesh=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7]
                        ],
    )

    Let's say we distribute a global_tensor of shape (8,4) over the above DeviceMesh
    with a placements of [Shard(0), Shard(0)].
    The local shape and global offset will be as follows:
    rank0 -- local_shape:[1, 4], global_offset:[0, 0]
    rank1 -- local_shape:[1, 4], global_offset:[1, 0]
    rank2 -- local_shape:[1, 4], global_offset:[2, 0]
    rank5 -- local_shape:[1, 4], global_offset:[5, 0]
    rank3 -- local_shape:[1, 4], global_offset:[3, 0]
    rank4 -- local_shape:[1, 4], global_offset:[4, 0]
    rank6 -- local_shape:[1, 4], global_offset:[6, 0]
    rank7 -- local_shape:[1, 4], global_offset:[7, 0]

    Let's say we distribute a global_tensor of shape (2) over the above DeviceMesh with
    a placements of [Shard(0)]. We will not have non-empty local tensor for all the ranks.
    The local shape and global offset will be as follows:
    rank0 -- local_shape:[1,], global_offset:[0,]
    rank1 -- local_shape:[1,], global_offset:[1,]
    rank2 -- local_shape:[0,], global_offset:[2,]
    rank5 -- local_shape:[0,], global_offset:[2,]
    rank3 -- local_shape:[0,], global_offset:[2,]
    rank4 -- local_shape:[0,], global_offset:[2,]
    rank6 -- local_shape:[0,], global_offset:[2,]
    rank7 -- local_shape:[0,], global_offset:[2,]
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ((), ())
    else:
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)

        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                assert shard_dim < len(
                    local_shape
                ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )

                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset

                # On a given dimension, if the local_offset[shard_dim] is smaller than global_offset[shard_dim],
                # it means that this dimension has been already sharded in previous placement.
                # Therefore, we cannot simply replace the global_offset[shard_dim] with local_offset[shard_dim].
                # Instead, for the given shard_dim, we need to add local_offset[shard_dim] to existing global_offset[shard_dim].
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]

        return tuple(local_shape), tuple(global_offset)


def compute_global_tensor_info(
    tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[List[int], List[int]]:
    """
    Compute the global size and stride of a DTensor from the given local tensor.
    The local size is multiplited by `world_size` per Sharding dim.
    The local stride is multiplited by `world_size` per Sharding dim, as long as the
    dimension is outside sharding dim.

    For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).
    If the DTensor placements are [Shard(2)] and world_size is 2;
    then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Local tensor which DTensor will be constructed from.
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: A List of int which specifies the size of DTensor which build
            on top of the local tensor.
        tensor_stride: A List of int which specifies the stride of DTensor.
    """
    tensor_shape = list(tensor.size())
    tensor_stride = list(tensor.stride())
    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)
        if placement.is_shard():
            shard_placement = cast(Shard, placement)
            if shard_placement.dim < 0:
                raise AssertionError(
                    "Shard placements should have negative dims normalized in "
                    f"the user-facing APIs: {shard_placement}"
                )
            shard_dim = shard_placement.dim

            assert (
                shard_dim < tensor.ndim
            ), f"Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim} for placement number {idx}."

            local_dim_size = tensor_shape[shard_dim]
            tensor_shape[shard_dim] = local_dim_size * mesh_dim_size

            # recover tensor stride by modifying the stride that larger than
            # the current stride on the shard_dim
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    # rescale the stride by the shard size
                    tensor_stride[i] = tensor_stride[i] * mesh_dim_size
        elif not isinstance(placement, (Replicate, Partial)):
            raise RuntimeError(f"placement type {type(placement)} not supported!")
    return tensor_shape, tensor_stride


def try_find_mesh_from_args(
    op_call: torch._ops.OpOverload, args: Sequence[object]
) -> DeviceMesh:
    """
    Find the device mesh object from args.
    It returns None if no mesh is found.
    NOTE: we can optimize this search if needed
    """
    for arg in args:
        if isinstance(arg, (dtensor.DTensor, DTensorSpec)):
            return arg.device_mesh
        elif (
            isinstance(arg, (list, tuple))
            and len(arg) > 0
            and isinstance(arg[0], (dtensor.DTensor, DTensorSpec))
        ):
            return arg[0].device_mesh

    raise ValueError(f"Cannot find device mesh from args for op : {op_call}.")


def compute_local_stride(
    global_stride: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[int, ...]:
    """
    Compute the stride of a local tensor shard, given the global stride of the DTensor.
    NOTE: Currently this function is assuming the DTensor is evenly shardable.
    """
    stride_divisors = [1] * len(global_stride)
    for mesh_idx, p in enumerate(placements):
        if p.is_shard():
            i = cast(Shard, p).dim
            # tensor dimension i is sharded on mesh dimension mesh_idx,
            # so we need to divide all the strides larger than stride[i]
            # (by the submesh size)
            for j in range(len(global_stride)):
                if global_stride[j] > global_stride[i]:
                    stride_divisors[j] *= mesh.size(mesh_idx)
    return tuple(
        global_stride[i] // stride_divisors[i] for i in range(len(global_stride))
    )


def compute_padded_and_unpadded_local_shape(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    This util computes the padded and unpadded local shape of a DTensor. The padded shape is computed by considering
    applying padding to the global tensor such that each padded shard would have the exact same shape across all ranks.
    This means sharding happens after padding.

    This differs from `compute_local_shape`. The local shape from `compute_local_shape` considers padding after sharding,
    meaning padding is applied on each placements instead of globally. Therefore, the local shape on each shard
    after padding could be different. The local shape returned from `compute_local_shape` is different from the
    unpadded local shape.

    Padded and unpadded local shape could be the same depending on whether padding is needed on the current shard.
    """

    # Calculate globally how many chunks a given tensor dim will have globally.
    num_chunks_by_dim = [1 for _ in enumerate(global_shape)]
    for mesh_idx, placement in enumerate(placements):
        if placement.is_shard():
            tensor_dim = placement.dim  # type: ignore[attr-defined]
            mesh_dim_size = mesh.size(mesh_idx)
            num_chunks_by_dim[tensor_dim] *= mesh_dim_size

    full_shard_size, cur_unpadded_shard_size = [], []
    for size_on_dim, num_chunks in zip(global_shape, num_chunks_by_dim):
        if num_chunks == 1:
            # This means no sharding is happening on the ith dimension of the global tensor.
            # Therefore, the padded and unpadded size of the ith dimension is the same as global_shape[i].
            full_shard_size.append(size_on_dim)
            cur_unpadded_shard_size.append(size_on_dim)
        else:
            # Calculate the full chunk size and the number of full chunks on a given tensor dim
            full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks
            num_full_chunks = size_on_dim // full_chunk_size
            tail_chunk_size = size_on_dim % full_chunk_size
            full_shard_size.append(full_chunk_size)

            # We can't use get_coordinate() here because get_coordinate() returns the ith shard on each
            # mesh dimension. When we move to global padding, we need to know the index of the shard on the tensor
            # dimension, because the ith tensor dimension could be sharded multiple times on different mesh dimension.
            # TODO: we would need a `get_coordinate_on_tensor_dim()` API to calculate this.
            # `get_rank()` would only work for 1D and 2D scenario.
            cur_chunk = mesh.get_rank()

            # If the index of cur chunk is smaller than num_full_chunks,
            # this means cur_chunk would be a full chunk on the given tensor dimension.
            if cur_chunk < num_full_chunks:
                cur_unpadded_shard_size.append(full_chunk_size)
            # If the index of cur_chunk is num_full_chunks and the tail_chunk_size is not 0,
            # this means the cur_chunk is the non-empty tail chunk.
            # There should be only 1 non-empty tail chunk.
            # For example, shard [1, 1, 1, 1, 1] to 4 chunks, we would have [1, 1], [1, 1], [1].
            # The third shard is a non-empty tail chunk and the last shard is an empty chunk.
            elif cur_chunk == num_full_chunks and tail_chunk_size != 0:
                cur_unpadded_shard_size.append(tail_chunk_size)
            # Otherwise, the cur_chunk is an empty chunk on the tensor_dim. There could be more than 1 empty chunks.
            # For example, chunk a tensor([1, 1]) into 4 chunks, the last two chunks would be empty.
            else:
                cur_unpadded_shard_size.append(0)

    return tuple(full_shard_size), tuple(cur_unpadded_shard_size)


def compute_padding_size(
    padded_size: Sequence[int], unpadded_size: Sequence[int]
) -> Tuple[int]:
    """
    Given the padded and unpadded shape of a tensor, this util returns a list of padding needed to make
    the unpadded tensor to be the same shape of padded tensor. The pad_size has the same length of padded_size and
    unpadded_size, in which the length equals to the number of dimensions of the tensor to be padded.

    padding_size[i] is the length of padding needed on the i-th tensor dimension.
    """
    assert len(padded_size) == len(unpadded_size)
    padding_size = []
    for padded_size_on_dim, unpadded_size_on_dim in zip(padded_size, unpadded_size):
        padding_size.append(padded_size_on_dim - unpadded_size_on_dim)
    return tuple(padding_size)
