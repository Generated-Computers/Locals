# The MTP layer's KV is one more entry in each request's cache list

The MTP layer (issue #18) attends over a KV of its own, one entry per position. That KV has to go wherever the
main model's cache goes: the prompt cache, the disk cache, checkpoints, reply state, and rows that join or leave
the batch. We made it one more entry at the end of the request's cache list, so with MTP on, the list has one
entry more than the model has layers. Nearly every place that copies, saves, merges, filters or trims the cache
already handles every entry of the list, so the entry goes along without any change there.

## Considered options

- **A KV that lives only while a request runs alone, empty at the start of each stretch.** It needs no change
  outside the solo path, but it is empty after every prompt-cache hit, and agents nearly always start from one.
  In the gap test (`docs/measurements/mtp/`, `results/gap_zwyrol.txt`), a layer that did not see the prompt got
  76.9% of first drafts right on the tool calls instead of 96.3%. On the issue's cost model that puts one draft
  below today's speed.

## Consequences

- Code that zips the cache with the model's layers stops at the last layer and never feeds the entry, so every
  place the main model runs fills it, or leaves it out of the list (the single-request engine, and Qwenputer,
  whose rows park and take no drafts).
- Disk-cache files saved before have one entry fewer and still load, so the disk-cache folder name carries an MTP
  mark.
- Each position costs an eighth more KV memory on Qwen3.5-4B: one more attention layer next to the main model's 8.
