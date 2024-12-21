#include "pdma_helpers.h"
#include "mchp-dma-proxy.h"

#include <stdio.h>


uint64_t
get_phy_addr(char* dev) {
    char bufinfo[128];
    uint64_t phy_addr;
    FILE*    fp;

    printf("Getting address of %s \n", &dev[strlen("/dev/")]);
    snprintf(bufinfo, sizeof(bufinfo), "/sys/class/u-dma-buf/%s/phys_addr", &dev[strlen("/dev/")]);
    printf("Getting address of %s \n", bufinfo);

    fp = fopen(bufinfo, "r");
    fscanf(fp, "%lx", &phy_addr);
    fclose(fp);

    printf("Phy addr of %s is 0x%lX\n", bufinfo, phy_addr);

    return phy_addr;
}


int32_t pdma_ch_open(){
	char channel_name[64] = {0};
    const char* dma_channel_names[] = { "dma-proxy0", /* add unique channel names here */ };
    int32_t chn_fd;
    snprintf(channel_name, sizeof(channel_name) - sizeof("/dev/"), "/dev/%s", dma_channel_names[0]);
    chn_fd = open(channel_name, O_RDWR);
	return chn_fd;
}

int32_t pdma_ch_cpy(uint64_t destbuf,  uint64_t srcbuf, size_t n, int32_t chn_fd){
	struct mpfs_dma_proxy_channel_config channel_config_t;
    channel_config_t.src = srcbuf;
    channel_config_t.dst = destbuf;
    channel_config_t.length = n;

    //printf("0x%lX == %.4f MB ==> 0x%lX \n",channel_config_t.src,n/(1024.0*1024),channel_config_t.dst);

    if (ioctl(chn_fd, MPFS_DMA_PROXY_START_XFER, &channel_config_t) != 0) {
        printf("transfer not started");
		return -1;
    }

    if (ioctl(chn_fd, MPFS_DMA_PROXY_FINISH_XFER, &proxy_status_e) != 0) {
        printf("transfer not finixhed");
		return -1;
    }

    return 0;
}

int32_t pdma_ch_close(int32_t chn_fd){
	if (close(chn_fd) != 0) {
		printf("failed to close");
        return -1;
    }
	return 0;
}

