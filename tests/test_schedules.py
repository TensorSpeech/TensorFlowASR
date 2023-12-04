import matplotlib.pyplot as plt

from tensorflow_asr.optimizers.schedules import CyclicTransformerSchedule, TransformerSchedule


def test_transformer_schedule():
    sched = TransformerSchedule(dmodel=176, scale=10.0, warmup_steps=10000, max_lr="0.05/(176**0.5)", min_lr=None)
    sched2 = CyclicTransformerSchedule(dmodel=320, step_size=10000, warmup_steps=15000, max_lr=0.0025)
    lrs = [sched(i).numpy() for i in range(100000)]
    print(lrs[:100])
    plt.plot(lrs)
    plt.show()
    lrs = [sched2(i).numpy() for i in range(100000)]
    print(lrs[:100])
    plt.plot(lrs)
    plt.show()
