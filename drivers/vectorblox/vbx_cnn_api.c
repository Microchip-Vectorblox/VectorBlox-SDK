#include "vbx_cnn_api.h"

static const int CTRL_OFFSET = 0;
static const int ERR_OFFSET = 1;
static const int ELF_OFFSET = 2;
static const int MODEL_OFFSET = 4;
static const int IO_OFFSET = 6;

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

static vbx_cnn_t the_cnn;
vbx_cnn_t *vbx_cnn_init(volatile void *ctrl_reg_addr, void *instruction_blob) {
  if (((uintptr_t)instruction_blob) & 1024 * 1024 * 2 - 1) {
    // instruction_blob must be aligned to a 2M boundary
    return NULL;
  }
  uint32_t checksum = fletcher32(instruction_blob, 2 * 1024 * 1024 - 4);
  uint32_t expected = ((uint32_t *)instruction_blob)[2 * 1024 * 1024 / 4 - 1];
  if (checksum != expected) {
    // firmware does not look correct. Perhaps it was loaded incorrectly.
    return NULL;
  }
  the_cnn.ctrl_reg = ctrl_reg_addr;
  the_cnn.instruction_blob = instruction_blob;
  // processor in reset:
  the_cnn.ctrl_reg[CTRL_OFFSET] = CTRL_REG_SOFT_RESET;
  // start processor:
  the_cnn.ctrl_reg[ELF_OFFSET] = (uintptr_t)instruction_blob & 0xFFFFFFFF;
  the_cnn.ctrl_reg[CTRL_OFFSET] = 0;
  the_cnn.initialized = 1;
  return &the_cnn;
}

vbx_cnn_err_e vbx_cnn_get_error_val(vbx_cnn_t *vbx_cnn) {
  return vbx_cnn->ctrl_reg[ERR_OFFSET];
}
int vbx_cnn_model_start(vbx_cnn_t *vbx_cnn, model_t *model,
                        vbx_cnn_io_ptr_t io_buffers[]) {
  vbx_cnn_state_e state = vbx_cnn_get_state(vbx_cnn);
  if (state != FULL && state != ERROR) {
    // wait until start bit is low before starting next model
    while (vbx_cnn->ctrl_reg[CTRL_OFFSET] & CTRL_REG_START)
      ;
    vbx_cnn->ctrl_reg[IO_OFFSET] = (uintptr_t)(io_buffers);
    vbx_cnn->ctrl_reg[MODEL_OFFSET] = (uintptr_t)model;

    // Start should be written with 1, other bits should be written
    // with zeros.
    vbx_cnn->ctrl_reg[CTRL_OFFSET] = CTRL_REG_START;
    return 0;
  } else {
    return -1;
  }
}

vbx_cnn_state_e vbx_cnn_get_state(vbx_cnn_t *vbx_cnn) {
  uint32_t ctrl_reg = vbx_cnn->ctrl_reg[CTRL_OFFSET];
  if (ctrl_reg & CTRL_REG_ERROR) {
    return (vbx_cnn_state_e)ERROR;
  }

  // if the error bit is not set,
  // we care about the combination of start/running/output_valid
  // to calculate the state
  ctrl_reg >>= 1;
  ctrl_reg &= 7;
  vbx_cnn_state_e state;
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
  int status = vbx_cnn->ctrl_reg[CTRL_OFFSET];
  if (status & CTRL_REG_OUTPUT_VALID) {
    // write 1 to clear output valid
    vbx_cnn->ctrl_reg[CTRL_OFFSET] = CTRL_REG_OUTPUT_VALID;
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
