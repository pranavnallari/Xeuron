// data.c
/*This file processes the recieved test data into workable arrays*/

#include "defs.h"

unsigned int read_uint32(FILE *file) {
    unsigned int value = 0;
    fread(&value, sizeof(value), 1, file);
    return (value >> 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0x0000FF00) | (value << 24);
}

float* load_mnist_images(const char *filename, unsigned int *num_images) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open image file");
        exit(1);
    }

    read_uint32(file);
    *num_images = read_uint32(file);
    unsigned int num_rows = read_uint32(file);
    unsigned int num_cols = read_uint32(file);

    if (num_rows != 28 || num_cols != 28) {
        fprintf(stderr, "Invalid image dimensions\n");
        exit(1);
    }

    float *images = (float *)malloc(*num_images * IMAGE_SIZE * sizeof(float));

    unsigned char *raw_images = (unsigned char *)malloc(*num_images * IMAGE_SIZE * sizeof(unsigned char));
    fread(raw_images, sizeof(unsigned char), *num_images * IMAGE_SIZE, file);

    for (unsigned int i = 0; i < *num_images * IMAGE_SIZE; ++i) {
        images[i] = raw_images[i] / 255.0f;
    }

    free(raw_images);
    fclose(file);
    return images;
}

unsigned char* load_mnist_labels(const char *filename, unsigned int *num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open label file");
        exit(1);
    }

    read_uint32(file);
    *num_labels = read_uint32(file);
    unsigned char *labels = (unsigned char *)malloc(*num_labels * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), *num_labels, file);

    fclose(file);
    return labels;
}

void set_input_value(Layer *input_layer, float *image) {
    for (unsigned i = 0; i < input_layer->num_neurons; ++i) {
        input_layer->neurons[i].value = image[i] / 255.0f;
    }
}