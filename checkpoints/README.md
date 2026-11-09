# Training records

Each folder holds the record of one training run. `config_used.yaml` is the resolved
configuration, and `metrics.jsonl` has one JSON line per epoch with the training loss, the epoch
time and the CER, NED, WER and line accuracy on the validation data, Test-I and Test-II, evaluated
while the model trained. The trained weights (`best.pt` and `last.pt`, 47 to 378 MB each) are not
included in this repository.

Two runs are incomplete. `parseq_small_s1` stopped after 5 of its 80 epochs, and `trocr_small_s1`
failed at start-up because of a library incompatibility, so it has a configuration but no log.
