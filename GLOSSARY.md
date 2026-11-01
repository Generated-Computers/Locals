# Locals

A local inference server for Apple Silicon. This fork also lets Qwen3.5 and Qwen3-VL models read images.

## Language

**Image prompt**:
A prompt that holds one or more images, each written as its own run of image tokens, read with every token at its image-aware position.
_Avoid_: vision prompt, embedded prompt, image request

**Cache key**:
The token sequence a prompt is stored under in the prompt cache. For a text prompt it is the prompt itself; for an image prompt, each image's tokens are replaced by that image's identity, so two different images never share an entry. An image prompt with an image of unknown identity has no cache key and is not cached.
_Avoid_: lookup tokens, key tokens
