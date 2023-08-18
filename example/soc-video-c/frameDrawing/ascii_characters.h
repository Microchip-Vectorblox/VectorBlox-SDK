#ifndef ALPHABET_H
#define ALPHABET_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct character_t{
	char c;
	int width;
	int height;
	char* data;
};
extern struct character_t ascii_characters_blue[];
extern struct character_t ascii_characters_white[];
extern struct character_t ascii_characters_orange[];
extern struct character_t ascii_characters_red[];
extern struct character_t ascii_characters_green[];
extern const int character_bin_length;
#ifdef __cplusplus
}
#endif
#endif //ALPHABET_H
