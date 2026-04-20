### NOTE
**Make is stupid about CLI args:** extra words after the target are Make’s goals, not the script’s. For `train` with flags, run `.venv/bin/python src/logreg_train.py …` instead of `make train …`. Later we switch to use ``uv`` and wont have this problem.

----

### **`logreg_train`:** New flags (optimizer, learning rate, epochs, batch size for MBGD, plot loss). Full list: `.venv/bin/python src/logreg_train.py -h`.

- **+2 BONUS — SGD / MBGD:** `--optimizer sgd` (SGD) or `--optimizer mbgd` with optional `--batch-size` (only for minibatch GD).
- **+1 BONUS — Training loss plot:** `-pl` / `--plot-loss` saves loss curves to `model/training_loss.png`.

----

### **`confusion_matrix`:** 
- (+1 BONUS):** New script; ground-truth CSV is positional `truth`; plots the confusion matrix (predictions vs. truth).
