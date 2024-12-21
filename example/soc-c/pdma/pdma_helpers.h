#ifndef PDMA_HELPERS_H
#define PDMA_HELPERS_H

#include <stdint.h>
#include <string.h>


#include <sys/fcntl.h> 
#include <sys/stat.h>
#include <sys/ioctl.h>      
#include <unistd.h>     
#include <stdio.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C" {
#endif



uint64_t get_phy_addr(char* dev);

int32_t pdma_ch_open();
int32_t pdma_ch_cpy(uint64_t destbuf,  uint64_t srcbuf, size_t n, int32_t chn_fd);
int32_t pdma_ch_close(int32_t chn_fd);

#ifdef __cplusplus
}
#endif

#endif /* PDMA_HELPERS_H */