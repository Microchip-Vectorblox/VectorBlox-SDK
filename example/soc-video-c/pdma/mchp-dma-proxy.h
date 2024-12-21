// SPDX-License-Identifier: MIT
/*
 * DMA proxy example for the Microchip PolarFire SoC.
 *
 *  This example demonstrates usage of DMA proxy driver and benchmarks
 *  it's transaction speeds against memcpy.
 *
 * Copyright (c) 2022 Microchip Technology Inc. All rights reserved.
 */

#ifndef _MPFS_DMA_PROXY_H
#define _MPFS_DMA_PROXY_H

#define MPFS_DMA_PROXY_IOC_MAGIC    'u'

#define MPFS_DMA_PROXY_FINISH_XFER _IOW(MPFS_DMA_PROXY_IOC_MAGIC, \
					'1', enum mpfs_dma_proxy_status*)

#define MPFS_DMA_PROXY_START_XFER   _IOW(MPFS_DMA_PROXY_IOC_MAGIC, \
					 '2', struct mpfs_dma_proxy_channel_config*)

#define MPFS_DMA_PROXY_IOC_MAXNR 2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * struct mpfs_dma_proxy_channel_config  - dma config info
 * @src:    source dma phy address
 * @dst:    destination dma phy address
 * @length: size of data xfer in bytes
 */
struct mpfs_dma_proxy_channel_config {
	uint64_t src;
	uint64_t dst;
	size_t length;
};


enum mpfs_dma_proxy_status {
	PROXY_SUCCESS = 0,
	PROXY_BUSY,
	PROXY_TIMEOUT,
	PROXY_ERROR,
} proxy_status_e;

#ifdef __cplusplus
}
#endif

#endif /* _MPFS_DMA_PROXY_H */