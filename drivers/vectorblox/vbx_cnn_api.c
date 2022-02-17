#include "vbx_cnn_api.h"

#define debug(a) printf("%s:%d %s=%d\n",__FILE__,__LINE__,#a,(int)(uintptr_t)(a))
#define debugl(a) printf("%s:%d %s=%lld\n",__FILE__,__LINE__,#a,(intptr_t)(a))
#define debugx(a) printf("%s:%d %s=0x%08x\n",__FILE__,__LINE__,#a,(unsigned)(uintptr_t)(a))
#define debugp(a) printf("%s:%d %s=%p\n",__FILE__,__LINE__,#a,(void*)(a))
#define debugs(a) printf("%s:%d %s=%s\n",__FILE__,__LINE__,#a,(a))

#if 0
//SHOULD probably use this, but it wasn't working properly when I tried it JDV
#define read_register(a,offset)  __atomic_load_n((a)+(offset),__ATOMIC_SEQ_CST)
#define write_register(a,offset,val)  __atomic_store_n((a)+(offset),val,__ATOMIC_SEQ_CST)
#elif SPLASHKIT_PCIE
#include "pcie.h"
#include <stdio.h>
#define read_register(a,offset)  read_fpga_word((uintptr_t)(a+offset))
#define write_register(a,offset,val)  write_fpga_word((uintptr_t)(a+offset),val)
#else
#define read_register(a,offset)  (a)[(offset)]
#define write_register(a,offset,val)  ((a)[(offset)]=(val))

#endif
static const int CTRL_OFFSET = 0;
static const int ERR_OFFSET = 1;
static const int ELF_OFFSET = 2;
static const int MODEL_OFFSET = 4;
static const int IO_OFFSET = 6;
//static const int VERSION_OFFSET = 10;

static const int CTRL_REG_SOFT_RESET = 0x00000001;
static const int CTRL_REG_START = 0x00000002;
static const int CTRL_REG_RUNNING = 0x00000004;
static const int CTRL_REG_OUTPUT_VALID = 0x00000008;
static const int CTRL_REG_ERROR = 0x00000010;
static uint32_t fletcher32(const void *d, size_t len) {
  uint32_t c0, c1;
  unsigned int i;
  len /= sizeof(uint16_t);
  const uint16_t *data = d;
  for (c0 = c1 = 0; len >= 360; len -= 360) {
    for (i = 0; i < 360; ++i) {
      c0 = c0 + *data++;
      c1 = c1 + c0;
    }
    c0 = c0 % 65535;
    c1 = c1 % 65535;
  }
  for (i = 0; i < len; ++i) {
    c0 = c0 + *data++;
    c1 = c1 + c0;
  }
  c0 = c0 % 65535;
  c1 = c1 % 65535;
  return (c1 << 16 | c0);
}

#if VBX_SOC_DRIVER
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>


static uint64_t u64_from_attribute(const char* filename){
  FILE* fd = fopen(filename,"r");
  uint64_t ret;
  fscanf(fd,"0x%" PRIx64,&ret);
  fclose(fd);
  //printf("READ SYSFS attribute %s = 0x%"PRIx64"\n",filename,ret);
  return ret;
}

//find vbx uio in /sys/class/uio
//if ctrl_reg_addr is non-null then /sys/class/uio/uio%d/maps/map0/addr must match
static int find_uio_dev_num(void *ctrl_reg_addr){
  char buf[4096];
  char expected_name[]="vbx";
  for(int n=0;;n++){
    snprintf(buf,4096,"/sys/class/uio/uio%d/name",n);
    int name_fd = open(buf,O_RDONLY);
    if (name_fd>=0){
      read(name_fd,buf,4096);
      close(name_fd);
      if(strncmp(expected_name,buf,strlen(expected_name))==0){

        //name matches,
        //if ctrl_reg_addr is not NULL, make sure that matches as well.
        snprintf(buf,4096,"/sys/class/uio/uio%d/maps/map0/addr",n);
        uintptr_t addr =  u64_from_attribute(buf);
        if(ctrl_reg_addr && (uintptr_t)ctrl_reg_addr != addr){
          //name matched but address didn't, skip this device.
          continue;
        }
        return n;
      }
    }else{
      break;
    }
  }
  return -1;
}
static void* uio_mmap(int dev_num,int map_num){
  char filename[64];
  snprintf(filename,sizeof(filename),"/sys/class/uio/uio%d/maps/map%d/size",dev_num,map_num);
  int64_t size=u64_from_attribute(filename);
  snprintf(filename,sizeof(filename),"/dev/uio%d",dev_num);
  int fd = open(filename, O_RDWR);
  void* _ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                fd, map_num * sysconf(_SC_PAGESIZE));
  close(fd);
  return _ptr;
}
static void* mmap_vbx_registers(int dev_num){
  return uio_mmap(dev_num,0);

}
#define DMA_DEV "udmabuf-ddr-nc0"
static size_t uio_dma_size(){
  const char* filename="/sys/class/u-dma-buf/" DMA_DEV "/size";
  FILE* fd = fopen(filename,"r");
  uint64_t size;
  fscanf(fd,"%" PRId64,&size);
  fclose(fd);
  return size;
}
static void* mmap_vbx_dma(){
  int fd = open("/dev/"DMA_DEV, O_RDWR);

  size_t size = uio_dma_size();
  void* _ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  memset(_ptr,0,size);
  close(fd);

  return _ptr;
}
static uintptr_t uio_dma_phys_addr(){
  char filename[64];
  snprintf(filename,sizeof(filename),"/sys/class/u-dma-buf/"DMA_DEV"/phys_addr");
  uintptr_t addr=u64_from_attribute(filename);
  return addr;
}
static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
  return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}
//static inline void* phys_to_virt(vbx_cnn_t* vbx_cnn,void* phys){
//  return (char*)(phys) - vbx_cnn->dma_phys_trans_offset;
//}
#else
static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
	return virt;
}
#endif //!VBX_SOC_DRIVER

vbx_cnn_t *vbx_cnn_init(void *ctrl_reg_addr,void *instruction_blob) {
  vbx_cnn_t* the_cnn = (vbx_cnn_t*)malloc(sizeof(vbx_cnn_t));
#if VBX_SOC_DRIVER
  int uio_dev_num = find_uio_dev_num(ctrl_reg_addr);
  ctrl_reg_addr = (void*)mmap_vbx_registers(uio_dev_num);
  the_cnn->dma_buffer=mmap_vbx_dma(uio_dev_num);
  the_cnn->dma_phys_trans_offset = uio_dma_phys_addr(uio_dev_num)-(uintptr_t)the_cnn->dma_buffer;
  the_cnn->dma_buffer_end = the_cnn->dma_buffer + uio_dma_size(uio_dev_num)-1;
  vbx_allocate_dma_buffer(the_cnn,0x1FFFFFFF,0);
  void* dma_instr_blob = vbx_allocate_dma_buffer(the_cnn,VBX_INSTRUCTION_SIZE,21);
  memcpy(dma_instr_blob,instruction_blob,VBX_INSTRUCTION_SIZE);
  the_cnn->io_buffers = vbx_allocate_dma_buffer(the_cnn,MAX_IO_BUFFERS*sizeof(vbx_cnn_io_ptr_t), 3);
#elif SPLASHKIT_PCIE
  fpga_init();
  void* FPGA_DDR_ADDR=(void*)0x40000000;
  const size_t FPGA_DDR_SIZE=1<<30;//1G
  the_cnn->dma_buffer=FPGA_DDR_ADDR;
  the_cnn->dma_phys_trans_offset = 0;
  the_cnn->dma_buffer_end = the_cnn->dma_buffer + FPGA_DDR_SIZE-1;
  void* dma_instr_blob = vbx_allocate_dma_buffer(the_cnn,VBX_INSTRUCTION_SIZE,21);
  write_fpga((uintptr_t)virt_to_phys(the_cnn,dma_instr_blob),instruction_blob,VBX_INSTRUCTION_SIZE);
  the_cnn->io_buffers = vbx_allocate_dma_buffer(the_cnn, MAX_IO_BUFFERS * sizeof(vbx_cnn_io_ptr_t), 3);
#else
  void* dma_instr_blob = instruction_blob;
#endif //VBX_SOC_DRIVER
  if (((uintptr_t)virt_to_phys(the_cnn,dma_instr_blob)) & (1024 * 1024 * 2 - 1)) {
    // instruction_blob must be aligned to a 2M boundary
    return NULL;
  }
  uint32_t checksum = fletcher32(instruction_blob, 2 * 1024 * 1024 - 4);
  uint32_t* expected = ((uint32_t *)instruction_blob)+(2 * 1024 * 1024 / 4 - 1);
  if (checksum != *expected) {
    // firmware does not look correct. Perhaps it was loaded incorrectly.
    return NULL;
  }
  //set checksum to zero to reset orca stdout pointer
  *expected=0;
  the_cnn->ctrl_reg = ctrl_reg_addr;
  the_cnn->instruction_blob = dma_instr_blob;
  // processor in reset:
  write_register(the_cnn->ctrl_reg,CTRL_OFFSET,CTRL_REG_SOFT_RESET);
  // start processor:
  write_register(the_cnn->ctrl_reg,ELF_OFFSET,(uintptr_t)virt_to_phys(the_cnn,dma_instr_blob) & 0xFFFFFFFF);
  write_register(the_cnn->ctrl_reg,CTRL_OFFSET, 0);
  the_cnn->initialized = 1;
  the_cnn->debug_print_ptr=0;
  return the_cnn;
}


#define DEBUG_PRINT_SPOOL_SIZE (16*1024)
int vbx_cnn_get_debug_prints(vbx_cnn_t* vbx_cnn,char* buf,size_t max_chars){
#if SPLASHKIT_PCIE
  static char orca_std_out[DEBUG_PRINT_SPOOL_SIZE];
  read_fpga((uintptr_t)(vbx_cnn->instruction_blob) + VBX_INSTRUCTION_SIZE - DEBUG_PRINT_SPOOL_SIZE,
      orca_std_out,
      DEBUG_PRINT_SPOOL_SIZE);
#else
  volatile char* orca_std_out = (volatile char*)(vbx_cnn->instruction_blob);
  orca_std_out+=VBX_INSTRUCTION_SIZE-DEBUG_PRINT_SPOOL_SIZE;
#endif
  volatile int32_t* orca_pointer = (volatile int32_t*)(orca_std_out + 16*1024-4);
  size_t char_count=0;
  while(vbx_cnn->debug_print_ptr != *orca_pointer && char_count+1 <= max_chars){
    buf[char_count]=orca_std_out[vbx_cnn->debug_print_ptr];
    vbx_cnn->debug_print_ptr++;
    char_count++;
    if(vbx_cnn->debug_print_ptr >=DEBUG_PRINT_SPOOL_SIZE-4){
      vbx_cnn->debug_print_ptr=0;
    }
  }
  return (int)char_count;

}
#if VBX_SOC_DRIVER || SPLASHKIT_PCIE
void* vbx_allocate_dma_buffer(vbx_cnn_t* vbx_cnn,size_t request_size,size_t phys_alignment_bits){
  size_t request_size_aligned = (request_size+1023) & (~1023);
  int alignment = 1<<phys_alignment_bits;
  uintptr_t cur_phys = (uintptr_t)virt_to_phys(vbx_cnn,vbx_cnn->dma_buffer);
  int incr = alignment - ((uintptr_t)(cur_phys) % alignment);
  vbx_cnn->dma_buffer += (incr % alignment);

  if (vbx_cnn->dma_buffer + request_size_aligned > vbx_cnn->dma_buffer_end){
    return NULL;
  }
  void* ret = (void*)vbx_cnn->dma_buffer;
  vbx_cnn->dma_buffer+= request_size_aligned;
  return ret;
}
void* vbx_get_dma_pointer(vbx_cnn_t* vbx_cnn){
  return (void*)(vbx_cnn->dma_buffer);
}
#endif



vbx_cnn_err_e vbx_cnn_get_error_val(vbx_cnn_t *vbx_cnn) {
  return read_register(vbx_cnn->ctrl_reg,ERR_OFFSET);
}
int vbx_cnn_model_start(vbx_cnn_t *vbx_cnn, model_t *model,
                        vbx_cnn_io_ptr_t io_buffers[]) {
#if VBX_SOC_DRIVER
  size_t num_io_buffers = (model_get_num_inputs(model)+
                        model_get_num_outputs(model));
  for(int io=0;io<num_io_buffers;io++){
    vbx_cnn->io_buffers[io] = (vbx_cnn_io_ptr_t)virt_to_phys(vbx_cnn,(void*)io_buffers[io]);
  }
  io_buffers = vbx_cnn->io_buffers;
#elif SPLASHKIT_PCIE
    static uint8_t temp_buffer[64 * 1024];
    model_t* readable_model = (model_t*)temp_buffer;
    read_fpga((uintptr_t)model, readable_model, sizeof(temp_buffer));
    size_t num_io_buffers = (model_get_num_inputs(readable_model) +
            model_get_num_outputs(readable_model));
    write_fpga((uintptr_t)vbx_cnn->io_buffers, io_buffers, num_io_buffers * sizeof(vbx_cnn_io_ptr_t));
    io_buffers = vbx_cnn->io_buffers;
#endif
  vbx_cnn_state_e state = vbx_cnn_get_state(vbx_cnn);
  if (state != FULL && state != ERROR) {
    // wait until start bit is low before starting next model
    while (read_register(vbx_cnn->ctrl_reg,CTRL_OFFSET) & CTRL_REG_START)
      ;
    write_register(vbx_cnn->ctrl_reg,IO_OFFSET , (uint32_t)(uintptr_t)virt_to_phys(vbx_cnn,io_buffers));
    write_register(vbx_cnn->ctrl_reg,MODEL_OFFSET , (uint32_t)(uintptr_t)virt_to_phys(vbx_cnn,model));

    // Start should be written with 1, other bits should be written
    // with zeros.
    write_register(vbx_cnn->ctrl_reg,CTRL_OFFSET,CTRL_REG_START);
    return 0;
  } else {
    return -1;
  }
}

vbx_cnn_state_e vbx_cnn_get_state(vbx_cnn_t *vbx_cnn) {
  uint32_t ctrl_reg = read_register(vbx_cnn->ctrl_reg,CTRL_OFFSET);
  if (ctrl_reg & CTRL_REG_ERROR) {
    return (vbx_cnn_state_e)ERROR;
  }

  // if the error bit is not set,
  // we care about the combination of start/running/output_valid
  // to calculate the state
  ctrl_reg >>= 1;
  ctrl_reg &= 7;
  vbx_cnn_state_e state=ERROR;
  switch (ctrl_reg) {
  case 0:
    state = READY;
    break;
  case 1:
    state = RUNNING_READY;
    break;
  case 2:
    state = RUNNING_READY;
    break;
  case 3:
    state = RUNNING;
    break;
  case 4:
    state = READY;
    break;
  case 5:
    state = RUNNING_READY;
    break;
  case 6:
    state = RUNNING_READY;
    break;
  case 7:
    state = FULL;
    break;
  }
  return state;
}

int vbx_cnn_model_poll(vbx_cnn_t *vbx_cnn) {
  int status = read_register(vbx_cnn->ctrl_reg,CTRL_OFFSET);
  if (status & CTRL_REG_OUTPUT_VALID) {
    // write 1 to clear output valid
    write_register(vbx_cnn->ctrl_reg,CTRL_OFFSET,CTRL_REG_OUTPUT_VALID);
    return 0;
  }
  if ((status & CTRL_REG_START) || (status & CTRL_REG_RUNNING)) {
    return 1;
  }
  if (status & CTRL_REG_ERROR) {
    return -1;
  }
  return -2;
}
