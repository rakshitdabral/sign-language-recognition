from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

acc = EventAccumulator(
    "Logs/train/events.out.tfevents.1708611375.VINCETQ.17324.0.v2"
)
acc.Reload()
print(acc.Tags())

