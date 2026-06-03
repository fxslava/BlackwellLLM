/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier:
 * LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "nvcomp/shared_types.h"

namespace nvcomp
{

/**
 * @brief Format specification for ANS compression
 */
struct ANSFormatSpecHeader
{
  // Empty for now
};

/**
 * @brief Format specification for Bitcomp compression
 */
struct BitcompFormatSpecHeader
{
  /**
   * @brief Bitcomp algorithm options.
   *
   * - 0 : Default algorithm, usually gives the best compression ratios
   * - 1 : "Sparse" algorithm, works well on sparse data (with lots of zeroes).
   *        and is usually a faster than the default algorithm.
   */
  int algorithm;
  /**
   * @brief One of nvcomp's possible data types
   */
  nvcompType_t data_type;
};

/**
 * @brief Format specification for Cascaded compression
 */
struct CascadedFormatSpecHeader
{
  /**
   * @brief The size of each internal chunk of data to decompress independently
   * with
   *
   * Cascaded compression. The value should be in the range of [512, 16384]
   * depending on the datatype of the input and the shared memory size of
   * the GPU being used.  This is not the size of chunks passed into the API.
   * Recommended size is 4096.
   *
   * @note Not currently used and a default of 4096 is just used.
   */
  size_t internal_chunk_bytes;
  /**
   * @brief The datatype used to define the bit-width for compression
   */
  nvcompType_t type;
  /**
   * @brief The number of Run Length Encodings to perform.
   */
  int num_RLEs;
  /**
   * @brief The number of Delta Encodings to perform.
   */
  int num_deltas;
  /**
   * @brief Whether or not to bitpack the final layers.
   */
  int use_bp;
};

/**
 * @brief Format specification for Deflate compression
 */
struct DeflateFormatSpecHeader
{
  /**
   * @brief Compression algorithm to use.
   *
   * - 0: highest-throughput, entropy-only compression (use for symmetric
   * compression/decompression performance)
   * - 1: high-throughput, low compression ratio (default)
   * - 2: medium-throughput, medium compression ratio, beat Zlib level 1 on the
   * compression ratio
   * - 3: placeholder for further compression level support, will fall into
   * MEDIUM_COMPRESSION at this point
   * - 4: lower-throughput, higher compression ratio, beat Zlib level 6 on the
   * compression ratio
   * - 5: lowest-throughput, highest compression ratio
   */
  int algorithm;
};

/**
 * @brief Format specification for GDeflate compression
 */
struct GdeflateFormatSpecHeader
{
  /**
   * @brief Compression algorithm to use.
   *
   * - 0: highest-throughput, entropy-only compression (use for symmetric
   * compression/decompression performance)
   * - 1: high-throughput, low compression ratio (default)
   * - 2: medium-throughput, medium compression ratio, beat Zlib level 1 on the
   * compression ratio
   * - 3: placeholder for further compression level support, will fall into
   * MEDIUM_COMPRESSION at this point
   * - 4: lower-throughput, higher compression ratio, beat Zlib level 6 on the
   * compression ratio
   * - 5: lowest-throughput, highest compression ratio
   */
  int algorithm;
};

/**
 * @brief Format specification for Gzip compression
 */
struct GzipFormatSpecHeader
{
  // Empty for now
};

/**
 * @brief Format specification for LZ4 compression
 */
struct LZ4FormatSpecHeader
{
  /**
   * @brief LZ4 data type to use.
   */
  union {
    nvcompType_t data_type;
    unsigned char bytes[4];
  };
};

/**
 * @brief Format specification for Snappy compression
 */
struct SnappyFormatSpecHeader
{
  // Empty for now
};

/**
 * @brief Format specification for Zstd compression
 */
struct ZstdFormatSpecHeader
{
  // Empty for now
};

} // namespace nvcomp
