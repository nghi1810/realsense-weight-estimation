import re
import matplotlib.pyplot as plt

# =========================
# 📌 PASTE LOG VÀO ĐÂY
# =========================
log_text = """
Epoch 1/1000
313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 139ms/step - loss: 0.6951 - mae: 0.9852
Epoch 1: val_loss improved from None to 0.33167, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 1: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 89s 159ms/step - loss: 0.6320 - mae: 0.9153 - val_loss: 0.3317 - val_mae: 0.5530 - learning_rate: 1.0000e-04
Epoch 2/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 0.5466 - mae: 0.8125
Epoch 2: val_loss improved from 0.33167 to 0.30700, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 2: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.5309 - mae: 0.7955 - val_loss: 0.3070 - val_mae: 0.5449 - learning_rate: 1.0000e-04
Epoch 3/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 75ms/step - loss: 0.4819 - mae: 0.7349
Epoch 3: val_loss did not improve from 0.30700
313/313 ━━━━━━━━━━━━━━━━━━━━ 24s 78ms/step - loss: 0.4752 - mae: 0.7262 - val_loss: 0.3207 - val_mae: 0.5482 - learning_rate: 1.0000e-04
Epoch 4/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 0.4545 - mae: 0.6956
Epoch 4: val_loss improved from 0.30700 to 0.27427, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 4: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.4505 - mae: 0.6898 - val_loss: 0.2743 - val_mae: 0.4917 - learning_rate: 1.0000e-04
Epoch 5/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 0.4104 - mae: 0.6403
Epoch 5: val_loss did not improve from 0.27427
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 78ms/step - loss: 0.4104 - mae: 0.6406 - val_loss: 0.2969 - val_mae: 0.5237 - learning_rate: 1.0000e-04
Epoch 6/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 0.3989 - mae: 0.6321
Epoch 6: val_loss improved from 0.27427 to 0.26708, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 6: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.3934 - mae: 0.6238 - val_loss: 0.2671 - val_mae: 0.4953 - learning_rate: 1.0000e-04
Epoch 7/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 0.3697 - mae: 0.5975
Epoch 7: val_loss improved from 0.26708 to 0.23598, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 7: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.3738 - mae: 0.6006 - val_loss: 0.2360 - val_mae: 0.4344 - learning_rate: 1.0000e-04
Epoch 8/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 0.3591 - mae: 0.5859
Epoch 8: val_loss did not improve from 0.23598
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.3450 - mae: 0.5661 - val_loss: 0.2506 - val_mae: 0.4670 - learning_rate: 1.0000e-04
Epoch 9/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 0.3291 - mae: 0.5502
Epoch 9: val_loss did not improve from 0.23598
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.3254 - mae: 0.5441 - val_loss: 0.2440 - val_mae: 0.4379 - learning_rate: 1.0000e-04
Epoch 10/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 0.3123 - mae: 0.5220
Epoch 10: val_loss improved from 0.23598 to 0.18266, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 10: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.3146 - mae: 0.5270 - val_loss: 0.1827 - val_mae: 0.3473 - learning_rate: 1.0000e-04
Epoch 11/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.3087 - mae: 0.5221
Epoch 11: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.3050 - mae: 0.5206 - val_loss: 0.2312 - val_mae: 0.4401 - learning_rate: 1.0000e-04
Epoch 12/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2994 - mae: 0.5089
Epoch 12: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2944 - mae: 0.5056 - val_loss: 0.1960 - val_mae: 0.3761 - learning_rate: 1.0000e-04
Epoch 13/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2909 - mae: 0.5072
Epoch 13: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2853 - mae: 0.4984 - val_loss: 0.2140 - val_mae: 0.3902 - learning_rate: 1.0000e-04
Epoch 14/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2867 - mae: 0.4996
Epoch 14: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2820 - mae: 0.4917 - val_loss: 0.1909 - val_mae: 0.3547 - learning_rate: 1.0000e-04
Epoch 15/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2634 - mae: 0.4638
Epoch 15: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2720 - mae: 0.4782 - val_loss: 0.1829 - val_mae: 0.3445 - learning_rate: 1.0000e-04
Epoch 16/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2658 - mae: 0.4705
Epoch 16: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2664 - mae: 0.4721 - val_loss: 0.2056 - val_mae: 0.3709 - learning_rate: 1.0000e-04
Epoch 17/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2741 - mae: 0.4854
Epoch 17: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2704 - mae: 0.4800 - val_loss: 0.2093 - val_mae: 0.3875 - learning_rate: 1.0000e-04
Epoch 18/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2628 - mae: 0.4680
Epoch 18: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2591 - mae: 0.4613 - val_loss: 0.2096 - val_mae: 0.4022 - learning_rate: 1.0000e-04
Epoch 19/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2601 - mae: 0.4664
Epoch 19: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2528 - mae: 0.4556 - val_loss: 0.2023 - val_mae: 0.3648 - learning_rate: 1.0000e-04
Epoch 20/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2394 - mae: 0.4386
Epoch 20: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2444 - mae: 0.4483 - val_loss: 0.2438 - val_mae: 0.4222 - learning_rate: 1.0000e-04
Epoch 21/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2369 - mae: 0.4377
Epoch 21: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2364 - mae: 0.4354 - val_loss: 0.1917 - val_mae: 0.3637 - learning_rate: 1.0000e-04
Epoch 22/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2335 - mae: 0.4332
Epoch 22: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2359 - mae: 0.4372 - val_loss: 0.1858 - val_mae: 0.3557 - learning_rate: 1.0000e-04
Epoch 23/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2351 - mae: 0.4389
Epoch 23: val_loss did not improve from 0.18266
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2373 - mae: 0.4432 - val_loss: 0.1932 - val_mae: 0.3532 - learning_rate: 1.0000e-04
Epoch 24/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2375 - mae: 0.4426
Epoch 24: val_loss improved from 0.18266 to 0.17326, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 24: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.2378 - mae: 0.4432 - val_loss: 0.1733 - val_mae: 0.3475 - learning_rate: 1.0000e-04
Epoch 25/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2296 - mae: 0.4339
Epoch 25: val_loss improved from 0.17326 to 0.16138, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 25: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.2274 - mae: 0.4297 - val_loss: 0.1614 - val_mae: 0.3234 - learning_rate: 1.0000e-04
Epoch 26/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2338 - mae: 0.4405
Epoch 26: val_loss did not improve from 0.16138
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2319 - mae: 0.4380 - val_loss: 0.3258 - val_mae: 0.4933 - learning_rate: 1.0000e-04
Epoch 27/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2123 - mae: 0.4093
Epoch 27: val_loss did not improve from 0.16138
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2170 - mae: 0.4165 - val_loss: 0.1737 - val_mae: 0.3483 - learning_rate: 1.0000e-04
Epoch 28/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2298 - mae: 0.4397
Epoch 28: val_loss did not improve from 0.16138
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2206 - mae: 0.4230 - val_loss: 0.1836 - val_mae: 0.3638 - learning_rate: 1.0000e-04
Epoch 29/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2127 - mae: 0.4129
Epoch 29: val_loss did not improve from 0.16138
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2114 - mae: 0.4096 - val_loss: 0.3325 - val_mae: 0.5799 - learning_rate: 1.0000e-04
Epoch 30/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2091 - mae: 0.4077
Epoch 30: val_loss did not improve from 0.16138
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2098 - mae: 0.4096 - val_loss: 0.2020 - val_mae: 0.3830 - learning_rate: 1.0000e-04
Epoch 31/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2136 - mae: 0.4197
Epoch 31: val_loss improved from 0.16138 to 0.15619, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 31: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.2100 - mae: 0.4146 - val_loss: 0.1562 - val_mae: 0.3210 - learning_rate: 1.0000e-04
Epoch 32/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2072 - mae: 0.4090
Epoch 32: val_loss did not improve from 0.15619
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2055 - mae: 0.4088 - val_loss: 0.1659 - val_mae: 0.3401 - learning_rate: 1.0000e-04
Epoch 33/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1998 - mae: 0.3966
Epoch 33: val_loss did not improve from 0.15619
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2021 - mae: 0.4020 - val_loss: 0.1780 - val_mae: 0.3589 - learning_rate: 1.0000e-04
Epoch 34/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1979 - mae: 0.4003
Epoch 34: val_loss did not improve from 0.15619
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1973 - mae: 0.3995 - val_loss: 0.1835 - val_mae: 0.3582 - learning_rate: 1.0000e-04
Epoch 35/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2047 - mae: 0.4132
Epoch 35: val_loss did not improve from 0.15619
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.2027 - mae: 0.4109 - val_loss: 0.1795 - val_mae: 0.3568 - learning_rate: 1.0000e-04
Epoch 36/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.2011 - mae: 0.4077
Epoch 36: val_loss did not improve from 0.15619
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1986 - mae: 0.4046 - val_loss: 0.1804 - val_mae: 0.3668 - learning_rate: 1.0000e-04
Epoch 37/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1934 - mae: 0.3952
Epoch 37: val_loss improved from 0.15619 to 0.15041, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 37: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1922 - mae: 0.3939 - val_loss: 0.1504 - val_mae: 0.3214 - learning_rate: 1.0000e-04
Epoch 38/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1851 - mae: 0.3812
Epoch 38: val_loss did not improve from 0.15041
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1909 - mae: 0.3938 - val_loss: 0.1807 - val_mae: 0.3781 - learning_rate: 1.0000e-04
Epoch 39/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1868 - mae: 0.3918
Epoch 39: val_loss did not improve from 0.15041
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1907 - mae: 0.3957 - val_loss: 0.1546 - val_mae: 0.3282 - learning_rate: 1.0000e-04
Epoch 40/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1858 - mae: 0.3866
Epoch 40: val_loss did not improve from 0.15041
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1895 - mae: 0.3950 - val_loss: 0.1844 - val_mae: 0.3882 - learning_rate: 1.0000e-04
Epoch 41/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1771 - mae: 0.3719
Epoch 41: val_loss improved from 0.15041 to 0.13798, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 41: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1788 - mae: 0.3786 - val_loss: 0.1380 - val_mae: 0.3026 - learning_rate: 1.0000e-04
Epoch 42/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1766 - mae: 0.3770
Epoch 42: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1822 - mae: 0.3859 - val_loss: 0.1599 - val_mae: 0.3404 - learning_rate: 1.0000e-04
Epoch 43/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1792 - mae: 0.3829
Epoch 43: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1814 - mae: 0.3892 - val_loss: 0.1512 - val_mae: 0.3300 - learning_rate: 1.0000e-04
Epoch 44/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1740 - mae: 0.3751
Epoch 44: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1730 - mae: 0.3750 - val_loss: 0.1420 - val_mae: 0.3193 - learning_rate: 1.0000e-04
Epoch 45/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1753 - mae: 0.3773
Epoch 45: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1798 - mae: 0.3867 - val_loss: 0.1528 - val_mae: 0.3369 - learning_rate: 1.0000e-04
Epoch 46/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1663 - mae: 0.3662
Epoch 46: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1715 - mae: 0.3769 - val_loss: 0.1430 - val_mae: 0.3240 - learning_rate: 1.0000e-04
Epoch 47/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1667 - mae: 0.3710
Epoch 47: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1729 - mae: 0.3818 - val_loss: 0.1570 - val_mae: 0.3512 - learning_rate: 1.0000e-04
Epoch 48/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1749 - mae: 0.3840
Epoch 48: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1714 - mae: 0.3795 - val_loss: 0.1577 - val_mae: 0.3562 - learning_rate: 1.0000e-04
Epoch 49/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1734 - mae: 0.3866
Epoch 49: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1713 - mae: 0.3822 - val_loss: 0.1432 - val_mae: 0.3292 - learning_rate: 1.0000e-04
Epoch 50/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1643 - mae: 0.3704
Epoch 50: val_loss did not improve from 0.13798
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1668 - mae: 0.3754 - val_loss: 0.1433 - val_mae: 0.3307 - learning_rate: 1.0000e-04
Epoch 51/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1657 - mae: 0.3746
Epoch 51: val_loss improved from 0.13798 to 0.12843, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 51: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1671 - mae: 0.3763 - val_loss: 0.1284 - val_mae: 0.3033 - learning_rate: 1.0000e-04
Epoch 52/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1634 - mae: 0.3723
Epoch 52: val_loss did not improve from 0.12843
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1615 - mae: 0.3689 - val_loss: 0.1470 - val_mae: 0.3423 - learning_rate: 1.0000e-04
Epoch 53/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1664 - mae: 0.3789
Epoch 53: val_loss did not improve from 0.12843
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1647 - mae: 0.3761 - val_loss: 0.1322 - val_mae: 0.3157 - learning_rate: 1.0000e-04
Epoch 54/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1635 - mae: 0.3746
Epoch 54: val_loss did not improve from 0.12843
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1656 - mae: 0.3784 - val_loss: 0.1315 - val_mae: 0.3148 - learning_rate: 1.0000e-04
Epoch 55/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1539 - mae: 0.3598
Epoch 55: val_loss improved from 0.12843 to 0.12744, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 55: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1533 - mae: 0.3581 - val_loss: 0.1274 - val_mae: 0.3076 - learning_rate: 1.0000e-04
Epoch 56/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1617 - mae: 0.3754
Epoch 56: val_loss did not improve from 0.12744
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1611 - mae: 0.3746 - val_loss: 0.1325 - val_mae: 0.3202 - learning_rate: 1.0000e-04
Epoch 57/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1521 - mae: 0.3581
Epoch 57: val_loss did not improve from 0.12744
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1544 - mae: 0.3621 - val_loss: 0.1368 - val_mae: 0.3327 - learning_rate: 1.0000e-04
Epoch 58/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1505 - mae: 0.3586
Epoch 58: val_loss did not improve from 0.12744
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1551 - mae: 0.3663 - val_loss: 0.1521 - val_mae: 0.3617 - learning_rate: 1.0000e-04
Epoch 59/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1544 - mae: 0.3671
Epoch 59: val_loss did not improve from 0.12744
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1521 - mae: 0.3612 - val_loss: 0.1331 - val_mae: 0.3243 - learning_rate: 1.0000e-04
Epoch 60/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1553 - mae: 0.3687
Epoch 60: val_loss did not improve from 0.12744
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1547 - mae: 0.3664 - val_loss: 0.1308 - val_mae: 0.3205 - learning_rate: 1.0000e-04
Epoch 61/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1486 - mae: 0.3579
Epoch 61: val_loss did not improve from 0.12744
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1492 - mae: 0.3584 - val_loss: 0.1332 - val_mae: 0.3288 - learning_rate: 1.0000e-04
Epoch 62/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1500 - mae: 0.3617
Epoch 62: val_loss improved from 0.12744 to 0.12644, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 62: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1507 - mae: 0.3624 - val_loss: 0.1264 - val_mae: 0.3174 - learning_rate: 1.0000e-04
Epoch 63/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1439 - mae: 0.3511
Epoch 63: val_loss did not improve from 0.12644
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1456 - mae: 0.3539 - val_loss: 0.1322 - val_mae: 0.3271 - learning_rate: 1.0000e-04
Epoch 64/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1492 - mae: 0.3628
Epoch 64: val_loss did not improve from 0.12644
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1468 - mae: 0.3584 - val_loss: 0.1326 - val_mae: 0.3301 - learning_rate: 1.0000e-04
Epoch 65/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1446 - mae: 0.3518
Epoch 65: val_loss did not improve from 0.12644
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1443 - mae: 0.3519 - val_loss: 0.1401 - val_mae: 0.3476 - learning_rate: 1.0000e-04
Epoch 66/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1472 - mae: 0.3657
Epoch 66: val_loss improved from 0.12644 to 0.11913, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 66: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1465 - mae: 0.3627 - val_loss: 0.1191 - val_mae: 0.3063 - learning_rate: 1.0000e-04
Epoch 67/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1520 - mae: 0.3710
Epoch 67: val_loss did not improve from 0.11913
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1445 - mae: 0.3565 - val_loss: 0.1195 - val_mae: 0.3059 - learning_rate: 1.0000e-04
Epoch 68/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1496 - mae: 0.3623
Epoch 68: val_loss did not improve from 0.11913
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1471 - mae: 0.3584 - val_loss: 0.1207 - val_mae: 0.3088 - learning_rate: 1.0000e-04
Epoch 69/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1418 - mae: 0.3538
Epoch 69: val_loss improved from 0.11913 to 0.11299, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 69: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1439 - mae: 0.3591 - val_loss: 0.1130 - val_mae: 0.2954 - learning_rate: 1.0000e-04
Epoch 70/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1459 - mae: 0.3612
Epoch 70: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1442 - mae: 0.3589 - val_loss: 0.1364 - val_mae: 0.3445 - learning_rate: 1.0000e-04
Epoch 71/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1462 - mae: 0.3628
Epoch 71: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1419 - mae: 0.3543 - val_loss: 0.1246 - val_mae: 0.3213 - learning_rate: 1.0000e-04
Epoch 72/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1361 - mae: 0.3436
Epoch 72: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1453 - mae: 0.3612 - val_loss: 0.1205 - val_mae: 0.3156 - learning_rate: 1.0000e-04
Epoch 73/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1376 - mae: 0.3473
Epoch 73: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1407 - mae: 0.3547 - val_loss: 0.1203 - val_mae: 0.3169 - learning_rate: 1.0000e-04
Epoch 74/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1361 - mae: 0.3472
Epoch 74: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1376 - mae: 0.3501 - val_loss: 0.1371 - val_mae: 0.3427 - learning_rate: 1.0000e-04
Epoch 75/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1412 - mae: 0.3607
Epoch 75: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1398 - mae: 0.3576 - val_loss: 0.1248 - val_mae: 0.3186 - learning_rate: 1.0000e-04
Epoch 76/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1427 - mae: 0.3613
Epoch 76: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1386 - mae: 0.3533 - val_loss: 0.1387 - val_mae: 0.3455 - learning_rate: 1.0000e-04
Epoch 77/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1385 - mae: 0.3528
Epoch 77: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1377 - mae: 0.3524 - val_loss: 0.1770 - val_mae: 0.3983 - learning_rate: 1.0000e-04
Epoch 78/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1372 - mae: 0.3515
Epoch 78: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1370 - mae: 0.3519 - val_loss: 0.1172 - val_mae: 0.3110 - learning_rate: 1.0000e-04
Epoch 79/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1318 - mae: 0.3461
Epoch 79: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1323 - mae: 0.3458 - val_loss: 0.1177 - val_mae: 0.3124 - learning_rate: 1.0000e-04
Epoch 80/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1358 - mae: 0.3533
Epoch 80: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1369 - mae: 0.3543 - val_loss: 0.1135 - val_mae: 0.3035 - learning_rate: 1.0000e-04
Epoch 81/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1330 - mae: 0.3474
Epoch 81: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1309 - mae: 0.3425 - val_loss: 0.1134 - val_mae: 0.3024 - learning_rate: 1.0000e-04
Epoch 82/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1343 - mae: 0.3502
Epoch 82: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1395 - mae: 0.3608 - val_loss: 0.1161 - val_mae: 0.3106 - learning_rate: 1.0000e-04
Epoch 83/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1372 - mae: 0.3573
Epoch 83: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1342 - mae: 0.3514 - val_loss: 0.1179 - val_mae: 0.3171 - learning_rate: 1.0000e-04
Epoch 84/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1345 - mae: 0.3490
Epoch 84: ReduceLROnPlateau reducing learning rate to 6.999999823165126e-05.

Epoch 84: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1384 - mae: 0.3584 - val_loss: 0.1324 - val_mae: 0.3435 - learning_rate: 1.0000e-04
Epoch 85/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1330 - mae: 0.3526
Epoch 85: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1317 - mae: 0.3487 - val_loss: 0.1221 - val_mae: 0.3278 - learning_rate: 7.0000e-05
Epoch 86/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1265 - mae: 0.3391
Epoch 86: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1252 - mae: 0.3369 - val_loss: 0.1166 - val_mae: 0.3215 - learning_rate: 7.0000e-05
Epoch 87/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1257 - mae: 0.3360
Epoch 87: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1278 - mae: 0.3403 - val_loss: 0.1245 - val_mae: 0.3310 - learning_rate: 7.0000e-05
Epoch 88/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1302 - mae: 0.3462
Epoch 88: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1335 - mae: 0.3540 - val_loss: 0.1150 - val_mae: 0.3129 - learning_rate: 7.0000e-05
Epoch 89/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1346 - mae: 0.3529
Epoch 89: val_loss did not improve from 0.11299
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1326 - mae: 0.3499 - val_loss: 0.1195 - val_mae: 0.3245 - learning_rate: 7.0000e-05
Epoch 90/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1259 - mae: 0.3395
Epoch 90: val_loss improved from 0.11299 to 0.10957, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 90: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1263 - mae: 0.3389 - val_loss: 0.1096 - val_mae: 0.3028 - learning_rate: 7.0000e-05
Epoch 91/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1280 - mae: 0.3432
Epoch 91: val_loss did not improve from 0.10957
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1267 - mae: 0.3428 - val_loss: 0.1121 - val_mae: 0.3108 - learning_rate: 7.0000e-05
Epoch 92/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1280 - mae: 0.3454
Epoch 92: val_loss did not improve from 0.10957
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1244 - mae: 0.3372 - val_loss: 0.1146 - val_mae: 0.3186 - learning_rate: 7.0000e-05
Epoch 93/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1367 - mae: 0.3598
Epoch 93: val_loss did not improve from 0.10957
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1281 - mae: 0.3454 - val_loss: 0.1202 - val_mae: 0.3301 - learning_rate: 7.0000e-05
Epoch 94/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1305 - mae: 0.3545
Epoch 94: val_loss improved from 0.10957 to 0.10574, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 94: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1269 - mae: 0.3456 - val_loss: 0.1057 - val_mae: 0.2998 - learning_rate: 7.0000e-05
Epoch 95/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1248 - mae: 0.3390
Epoch 95: val_loss improved from 0.10574 to 0.10515, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 95: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1246 - mae: 0.3382 - val_loss: 0.1052 - val_mae: 0.3020 - learning_rate: 7.0000e-05
Epoch 96/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1227 - mae: 0.3357
Epoch 96: val_loss improved from 0.10515 to 0.10466, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 96: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1269 - mae: 0.3453 - val_loss: 0.1047 - val_mae: 0.2951 - learning_rate: 7.0000e-05
Epoch 97/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1249 - mae: 0.3435
Epoch 97: val_loss did not improve from 0.10466
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1212 - mae: 0.3351 - val_loss: 0.1161 - val_mae: 0.3248 - learning_rate: 7.0000e-05
Epoch 98/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1260 - mae: 0.3455
Epoch 98: val_loss did not improve from 0.10466
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1262 - mae: 0.3457 - val_loss: 0.1141 - val_mae: 0.3166 - learning_rate: 7.0000e-05
Epoch 99/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1235 - mae: 0.3388
Epoch 99: val_loss did not improve from 0.10466
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1244 - mae: 0.3413 - val_loss: 0.1249 - val_mae: 0.3465 - learning_rate: 7.0000e-05
Epoch 100/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1242 - mae: 0.3384
Epoch 100: val_loss improved from 0.10466 to 0.10372, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 100: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1236 - mae: 0.3375 - val_loss: 0.1037 - val_mae: 0.2952 - learning_rate: 7.0000e-05
Epoch 101/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1191 - mae: 0.3327
Epoch 101: val_loss did not improve from 0.10372
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1203 - mae: 0.3346 - val_loss: 0.1114 - val_mae: 0.3182 - learning_rate: 7.0000e-05
Epoch 102/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1259 - mae: 0.3478
Epoch 102: val_loss did not improve from 0.10372
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1236 - mae: 0.3440 - val_loss: 0.1125 - val_mae: 0.3126 - learning_rate: 7.0000e-05
Epoch 103/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1149 - mae: 0.3226
Epoch 103: val_loss did not improve from 0.10372
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1202 - mae: 0.3322 - val_loss: 0.1156 - val_mae: 0.3270 - learning_rate: 7.0000e-05
Epoch 104/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1206 - mae: 0.3349
Epoch 104: val_loss improved from 0.10372 to 0.10282, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 104: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1200 - mae: 0.3346 - val_loss: 0.1028 - val_mae: 0.2986 - learning_rate: 7.0000e-05
Epoch 105/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1259 - mae: 0.3422
Epoch 105: val_loss improved from 0.10282 to 0.09889, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 105: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1239 - mae: 0.3408 - val_loss: 0.0989 - val_mae: 0.2866 - learning_rate: 7.0000e-05
Epoch 106/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1212 - mae: 0.3384
Epoch 106: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1202 - mae: 0.3364 - val_loss: 0.1112 - val_mae: 0.3160 - learning_rate: 7.0000e-05
Epoch 107/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1166 - mae: 0.3273
Epoch 107: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1163 - mae: 0.3260 - val_loss: 0.1071 - val_mae: 0.3047 - learning_rate: 7.0000e-05
Epoch 108/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1191 - mae: 0.3371
Epoch 108: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1211 - mae: 0.3391 - val_loss: 0.1191 - val_mae: 0.3317 - learning_rate: 7.0000e-05
Epoch 109/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1207 - mae: 0.3379
Epoch 109: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1215 - mae: 0.3403 - val_loss: 0.1089 - val_mae: 0.3114 - learning_rate: 7.0000e-05
Epoch 110/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1220 - mae: 0.3353
Epoch 110: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1232 - mae: 0.3417 - val_loss: 0.1061 - val_mae: 0.3028 - learning_rate: 7.0000e-05
Epoch 111/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1163 - mae: 0.3294
Epoch 111: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1204 - mae: 0.3368 - val_loss: 0.1079 - val_mae: 0.3130 - learning_rate: 7.0000e-05
Epoch 112/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1268 - mae: 0.3509
Epoch 112: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1249 - mae: 0.3477 - val_loss: 0.1085 - val_mae: 0.3110 - learning_rate: 7.0000e-05
Epoch 113/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1133 - mae: 0.3245
Epoch 113: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1174 - mae: 0.3329 - val_loss: 0.1108 - val_mae: 0.3166 - learning_rate: 7.0000e-05
Epoch 114/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1272 - mae: 0.3502
Epoch 114: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1217 - mae: 0.3408 - val_loss: 0.1058 - val_mae: 0.3073 - learning_rate: 7.0000e-05
Epoch 115/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1230 - mae: 0.3440
Epoch 115: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1202 - mae: 0.3385 - val_loss: 0.1159 - val_mae: 0.3238 - learning_rate: 7.0000e-05
Epoch 116/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1215 - mae: 0.3437
Epoch 116: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1216 - mae: 0.3408 - val_loss: 0.1160 - val_mae: 0.3309 - learning_rate: 7.0000e-05
Epoch 117/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1168 - mae: 0.3317
Epoch 117: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1174 - mae: 0.3319 - val_loss: 0.1085 - val_mae: 0.3121 - learning_rate: 7.0000e-05
Epoch 118/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1175 - mae: 0.3354
Epoch 118: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1151 - mae: 0.3304 - val_loss: 0.1128 - val_mae: 0.3244 - learning_rate: 7.0000e-05
Epoch 119/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1191 - mae: 0.3384
Epoch 119: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1168 - mae: 0.3334 - val_loss: 0.1116 - val_mae: 0.3171 - learning_rate: 7.0000e-05
Epoch 120/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1222 - mae: 0.3419
Epoch 120: ReduceLROnPlateau reducing learning rate to 4.899999621557071e-05.

Epoch 120: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1186 - mae: 0.3344 - val_loss: 0.1191 - val_mae: 0.3384 - learning_rate: 7.0000e-05
Epoch 121/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1203 - mae: 0.3399
Epoch 121: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1202 - mae: 0.3414 - val_loss: 0.1134 - val_mae: 0.3230 - learning_rate: 4.9000e-05
Epoch 122/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1221 - mae: 0.3465
Epoch 122: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1134 - mae: 0.3294 - val_loss: 0.1146 - val_mae: 0.3173 - learning_rate: 4.9000e-05
Epoch 123/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1171 - mae: 0.3363
Epoch 123: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1162 - mae: 0.3345 - val_loss: 0.1131 - val_mae: 0.3278 - learning_rate: 4.9000e-05
Epoch 124/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1160 - mae: 0.3359
Epoch 124: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1140 - mae: 0.3301 - val_loss: 0.1058 - val_mae: 0.3117 - learning_rate: 4.9000e-05
Epoch 125/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1135 - mae: 0.3268
Epoch 125: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1115 - mae: 0.3263 - val_loss: 0.1036 - val_mae: 0.3063 - learning_rate: 4.9000e-05
Epoch 126/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1173 - mae: 0.3396
Epoch 126: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1194 - mae: 0.3428 - val_loss: 0.1080 - val_mae: 0.3112 - learning_rate: 4.9000e-05
Epoch 127/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1150 - mae: 0.3326
Epoch 127: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1141 - mae: 0.3308 - val_loss: 0.1167 - val_mae: 0.3294 - learning_rate: 4.9000e-05
Epoch 128/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1127 - mae: 0.3298
Epoch 128: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1134 - mae: 0.3307 - val_loss: 0.1024 - val_mae: 0.3030 - learning_rate: 4.9000e-05
Epoch 129/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1204 - mae: 0.3436
Epoch 129: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1179 - mae: 0.3386 - val_loss: 0.1037 - val_mae: 0.3059 - learning_rate: 4.9000e-05
Epoch 130/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1128 - mae: 0.3321
Epoch 130: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1128 - mae: 0.3296 - val_loss: 0.1057 - val_mae: 0.3027 - learning_rate: 4.9000e-05
Epoch 131/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1111 - mae: 0.3237
Epoch 131: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1095 - mae: 0.3220 - val_loss: 0.1127 - val_mae: 0.3258 - learning_rate: 4.9000e-05
Epoch 132/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1177 - mae: 0.3395
Epoch 132: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1184 - mae: 0.3409 - val_loss: 0.0992 - val_mae: 0.3009 - learning_rate: 4.9000e-05
Epoch 133/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1117 - mae: 0.3272
Epoch 133: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1150 - mae: 0.3339 - val_loss: 0.1095 - val_mae: 0.3156 - learning_rate: 4.9000e-05
Epoch 134/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1106 - mae: 0.3266
Epoch 134: val_loss did not improve from 0.09889
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1116 - mae: 0.3274 - val_loss: 0.1072 - val_mae: 0.3210 - learning_rate: 4.9000e-05
Epoch 135/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1148 - mae: 0.3357
Epoch 135: val_loss improved from 0.09889 to 0.09767, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 135: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1138 - mae: 0.3322 - val_loss: 0.0977 - val_mae: 0.2956 - learning_rate: 4.9000e-05
Epoch 136/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1127 - mae: 0.3287
Epoch 136: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1144 - mae: 0.3327 - val_loss: 0.1051 - val_mae: 0.3094 - learning_rate: 4.9000e-05
Epoch 137/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1141 - mae: 0.3311
Epoch 137: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1148 - mae: 0.3320 - val_loss: 0.1175 - val_mae: 0.3392 - learning_rate: 4.9000e-05
Epoch 138/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1141 - mae: 0.3339
Epoch 138: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1137 - mae: 0.3321 - val_loss: 0.1040 - val_mae: 0.3087 - learning_rate: 4.9000e-05
Epoch 139/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1141 - mae: 0.3342
Epoch 139: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1161 - mae: 0.3374 - val_loss: 0.1002 - val_mae: 0.2973 - learning_rate: 4.9000e-05
Epoch 140/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1120 - mae: 0.3281
Epoch 140: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1145 - mae: 0.3343 - val_loss: 0.1036 - val_mae: 0.3087 - learning_rate: 4.9000e-05
Epoch 141/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1109 - mae: 0.3304
Epoch 141: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1120 - mae: 0.3308 - val_loss: 0.1033 - val_mae: 0.3118 - learning_rate: 4.9000e-05
Epoch 142/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1141 - mae: 0.3338
Epoch 142: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1172 - mae: 0.3388 - val_loss: 0.1049 - val_mae: 0.3115 - learning_rate: 4.9000e-05
Epoch 143/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1137 - mae: 0.3315
Epoch 143: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1167 - mae: 0.3389 - val_loss: 0.1028 - val_mae: 0.3127 - learning_rate: 4.9000e-05
Epoch 144/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1033 - mae: 0.3156
Epoch 144: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1065 - mae: 0.3221 - val_loss: 0.1028 - val_mae: 0.3032 - learning_rate: 4.9000e-05
Epoch 145/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1144 - mae: 0.3334
Epoch 145: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1124 - mae: 0.3304 - val_loss: 0.0984 - val_mae: 0.2994 - learning_rate: 4.9000e-05
Epoch 146/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1113 - mae: 0.3291
Epoch 146: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1109 - mae: 0.3287 - val_loss: 0.1000 - val_mae: 0.3038 - learning_rate: 4.9000e-05
Epoch 147/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1164 - mae: 0.3409
Epoch 147: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1130 - mae: 0.3344 - val_loss: 0.1080 - val_mae: 0.3196 - learning_rate: 4.9000e-05
Epoch 148/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1226 - mae: 0.3514
Epoch 148: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1155 - mae: 0.3387 - val_loss: 0.1141 - val_mae: 0.3375 - learning_rate: 4.9000e-05
Epoch 149/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1121 - mae: 0.3346
Epoch 149: val_loss did not improve from 0.09767
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1098 - mae: 0.3275 - val_loss: 0.1007 - val_mae: 0.3077 - learning_rate: 4.9000e-05
Epoch 150/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1155 - mae: 0.3395
Epoch 150: val_loss improved from 0.09767 to 0.09611, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 150: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1110 - mae: 0.3295 - val_loss: 0.0961 - val_mae: 0.2922 - learning_rate: 4.9000e-05
Epoch 151/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1124 - mae: 0.3325
Epoch 151: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1106 - mae: 0.3301 - val_loss: 0.1011 - val_mae: 0.3015 - learning_rate: 4.9000e-05
Epoch 152/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1078 - mae: 0.3223
Epoch 152: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1126 - mae: 0.3317 - val_loss: 0.1074 - val_mae: 0.3183 - learning_rate: 4.9000e-05
Epoch 153/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1117 - mae: 0.3299
Epoch 153: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1127 - mae: 0.3315 - val_loss: 0.1039 - val_mae: 0.3073 - learning_rate: 4.9000e-05
Epoch 154/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1147 - mae: 0.3374
Epoch 154: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1156 - mae: 0.3404 - val_loss: 0.1021 - val_mae: 0.3042 - learning_rate: 4.9000e-05
Epoch 155/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1091 - mae: 0.3241
Epoch 155: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1119 - mae: 0.3294 - val_loss: 0.1287 - val_mae: 0.3484 - learning_rate: 4.9000e-05
Epoch 156/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1082 - mae: 0.3257
Epoch 156: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1088 - mae: 0.3256 - val_loss: 0.1026 - val_mae: 0.3136 - learning_rate: 4.9000e-05
Epoch 157/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1063 - mae: 0.3222
Epoch 157: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1059 - mae: 0.3211 - val_loss: 0.1011 - val_mae: 0.3079 - learning_rate: 4.9000e-05
Epoch 158/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1120 - mae: 0.3332
Epoch 158: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1083 - mae: 0.3253 - val_loss: 0.0997 - val_mae: 0.3041 - learning_rate: 4.9000e-05
Epoch 159/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1124 - mae: 0.3346
Epoch 159: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1140 - mae: 0.3367 - val_loss: 0.1011 - val_mae: 0.3049 - learning_rate: 4.9000e-05
Epoch 160/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1180 - mae: 0.3453
Epoch 160: val_loss did not improve from 0.09611
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1155 - mae: 0.3404 - val_loss: 0.0971 - val_mae: 0.2991 - learning_rate: 4.9000e-05
Epoch 161/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1157 - mae: 0.3361
Epoch 161: val_loss improved from 0.09611 to 0.09510, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 161: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1143 - mae: 0.3334 - val_loss: 0.0951 - val_mae: 0.2964 - learning_rate: 4.9000e-05
Epoch 162/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1125 - mae: 0.3356
Epoch 162: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1081 - mae: 0.3256 - val_loss: 0.0952 - val_mae: 0.2938 - learning_rate: 4.9000e-05
Epoch 163/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1166 - mae: 0.3417
Epoch 163: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1126 - mae: 0.3334 - val_loss: 0.1020 - val_mae: 0.3114 - learning_rate: 4.9000e-05
Epoch 164/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1159 - mae: 0.3414
Epoch 164: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1156 - mae: 0.3407 - val_loss: 0.1074 - val_mae: 0.3236 - learning_rate: 4.9000e-05
Epoch 165/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1081 - mae: 0.3257
Epoch 165: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1113 - mae: 0.3326 - val_loss: 0.1066 - val_mae: 0.3175 - learning_rate: 4.9000e-05
Epoch 166/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1099 - mae: 0.3335
Epoch 166: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1102 - mae: 0.3311 - val_loss: 0.0953 - val_mae: 0.2988 - learning_rate: 4.9000e-05
Epoch 167/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1061 - mae: 0.3213
Epoch 167: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1069 - mae: 0.3237 - val_loss: 0.1001 - val_mae: 0.3067 - learning_rate: 4.9000e-05
Epoch 168/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1042 - mae: 0.3163
Epoch 168: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1080 - mae: 0.3246 - val_loss: 0.1033 - val_mae: 0.3163 - learning_rate: 4.9000e-05
Epoch 169/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1064 - mae: 0.3231
Epoch 169: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1104 - mae: 0.3303 - val_loss: 0.0965 - val_mae: 0.2990 - learning_rate: 4.9000e-05
Epoch 170/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1054 - mae: 0.3208
Epoch 170: val_loss did not improve from 0.09510
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1080 - mae: 0.3282 - val_loss: 0.1013 - val_mae: 0.3089 - learning_rate: 4.9000e-05
Epoch 171/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1057 - mae: 0.3212
Epoch 171: val_loss improved from 0.09510 to 0.09339, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 171: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1092 - mae: 0.3272 - val_loss: 0.0934 - val_mae: 0.2939 - learning_rate: 4.9000e-05
Epoch 172/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1050 - mae: 0.3174
Epoch 172: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1065 - mae: 0.3220 - val_loss: 0.0992 - val_mae: 0.3074 - learning_rate: 4.9000e-05
Epoch 173/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1091 - mae: 0.3289
Epoch 173: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1092 - mae: 0.3298 - val_loss: 0.1047 - val_mae: 0.3187 - learning_rate: 4.9000e-05
Epoch 174/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1052 - mae: 0.3188
Epoch 174: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1062 - mae: 0.3224 - val_loss: 0.0936 - val_mae: 0.2954 - learning_rate: 4.9000e-05
Epoch 175/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1062 - mae: 0.3226
Epoch 175: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1087 - mae: 0.3267 - val_loss: 0.0976 - val_mae: 0.3036 - learning_rate: 4.9000e-05
Epoch 176/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1032 - mae: 0.3202
Epoch 176: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1061 - mae: 0.3252 - val_loss: 0.0989 - val_mae: 0.3099 - learning_rate: 4.9000e-05
Epoch 177/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1041 - mae: 0.3184
Epoch 177: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1063 - mae: 0.3227 - val_loss: 0.1000 - val_mae: 0.3110 - learning_rate: 4.9000e-05
Epoch 178/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1053 - mae: 0.3215
Epoch 178: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1053 - mae: 0.3227 - val_loss: 0.1017 - val_mae: 0.3118 - learning_rate: 4.9000e-05
Epoch 179/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1052 - mae: 0.3214
Epoch 179: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1081 - mae: 0.3285 - val_loss: 0.1003 - val_mae: 0.3088 - learning_rate: 4.9000e-05
Epoch 180/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1122 - mae: 0.3355
Epoch 180: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1098 - mae: 0.3297 - val_loss: 0.1094 - val_mae: 0.3290 - learning_rate: 4.9000e-05
Epoch 181/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1080 - mae: 0.3305
Epoch 181: val_loss did not improve from 0.09339
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1063 - mae: 0.3255 - val_loss: 0.1089 - val_mae: 0.3285 - learning_rate: 4.9000e-05
Epoch 182/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1097 - mae: 0.3301
Epoch 182: val_loss improved from 0.09339 to 0.09086, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 182: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1104 - mae: 0.3333 - val_loss: 0.0909 - val_mae: 0.2905 - learning_rate: 4.9000e-05
Epoch 183/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1097 - mae: 0.3308
Epoch 183: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1112 - mae: 0.3348 - val_loss: 0.1041 - val_mae: 0.3177 - learning_rate: 4.9000e-05
Epoch 184/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1047 - mae: 0.3239
Epoch 184: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1078 - mae: 0.3286 - val_loss: 0.1067 - val_mae: 0.3170 - learning_rate: 4.9000e-05
Epoch 185/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1108 - mae: 0.3331
Epoch 185: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1092 - mae: 0.3311 - val_loss: 0.1015 - val_mae: 0.3140 - learning_rate: 4.9000e-05
Epoch 186/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1066 - mae: 0.3271
Epoch 186: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1082 - mae: 0.3305 - val_loss: 0.1010 - val_mae: 0.3140 - learning_rate: 4.9000e-05
Epoch 187/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1051 - mae: 0.3237
Epoch 187: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1089 - mae: 0.3298 - val_loss: 0.1075 - val_mae: 0.3262 - learning_rate: 4.9000e-05
Epoch 188/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1082 - mae: 0.3284
Epoch 188: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1051 - mae: 0.3237 - val_loss: 0.0979 - val_mae: 0.3006 - learning_rate: 4.9000e-05
Epoch 189/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1127 - mae: 0.3365
Epoch 189: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1123 - mae: 0.3362 - val_loss: 0.1005 - val_mae: 0.3097 - learning_rate: 4.9000e-05
Epoch 190/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1082 - mae: 0.3313
Epoch 190: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1053 - mae: 0.3243 - val_loss: 0.0996 - val_mae: 0.3116 - learning_rate: 4.9000e-05
Epoch 191/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1094 - mae: 0.3310
Epoch 191: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1072 - mae: 0.3259 - val_loss: 0.0970 - val_mae: 0.3041 - learning_rate: 4.9000e-05
Epoch 192/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1082 - mae: 0.3310
Epoch 192: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1100 - mae: 0.3341 - val_loss: 0.0982 - val_mae: 0.3038 - learning_rate: 4.9000e-05
Epoch 193/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1062 - mae: 0.3273
Epoch 193: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1079 - mae: 0.3304 - val_loss: 0.0992 - val_mae: 0.3096 - learning_rate: 4.9000e-05
Epoch 194/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1055 - mae: 0.3264
Epoch 194: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1061 - mae: 0.3253 - val_loss: 0.0985 - val_mae: 0.3061 - learning_rate: 4.9000e-05
Epoch 195/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1051 - mae: 0.3250
Epoch 195: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1058 - mae: 0.3248 - val_loss: 0.0952 - val_mae: 0.3029 - learning_rate: 4.9000e-05
Epoch 196/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1061 - mae: 0.3253
Epoch 196: val_loss did not improve from 0.09086
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1029 - mae: 0.3197 - val_loss: 0.1085 - val_mae: 0.3222 - learning_rate: 4.9000e-05
Epoch 197/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1019 - mae: 0.3158
Epoch 197: val_loss improved from 0.09086 to 0.09067, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 197: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1040 - mae: 0.3216 - val_loss: 0.0907 - val_mae: 0.2927 - learning_rate: 4.9000e-05
Epoch 198/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1032 - mae: 0.3235
Epoch 198: val_loss did not improve from 0.09067
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1038 - mae: 0.3250 - val_loss: 0.1023 - val_mae: 0.3165 - learning_rate: 4.9000e-05
Epoch 199/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1078 - mae: 0.3318
Epoch 199: val_loss improved from 0.09067 to 0.08877, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 199: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1087 - mae: 0.3342 - val_loss: 0.0888 - val_mae: 0.2878 - learning_rate: 4.9000e-05
Epoch 200/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1126 - mae: 0.3380
Epoch 200: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1091 - mae: 0.3326 - val_loss: 0.0996 - val_mae: 0.3090 - learning_rate: 4.9000e-05
Epoch 201/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1071 - mae: 0.3307
Epoch 201: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1083 - mae: 0.3336 - val_loss: 0.0987 - val_mae: 0.3109 - learning_rate: 4.9000e-05
Epoch 202/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1031 - mae: 0.3210
Epoch 202: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1050 - mae: 0.3250 - val_loss: 0.0942 - val_mae: 0.2988 - learning_rate: 4.9000e-05
Epoch 203/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1101 - mae: 0.3353
Epoch 203: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1081 - mae: 0.3298 - val_loss: 0.0973 - val_mae: 0.3091 - learning_rate: 4.9000e-05
Epoch 204/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1031 - mae: 0.3207
Epoch 204: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1045 - mae: 0.3226 - val_loss: 0.0989 - val_mae: 0.3002 - learning_rate: 4.9000e-05
Epoch 205/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1097 - mae: 0.3348
Epoch 205: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1099 - mae: 0.3349 - val_loss: 0.0980 - val_mae: 0.3034 - learning_rate: 4.9000e-05
Epoch 206/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1121 - mae: 0.3384
Epoch 206: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1123 - mae: 0.3392 - val_loss: 0.0947 - val_mae: 0.3009 - learning_rate: 4.9000e-05
Epoch 207/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1157 - mae: 0.3476
Epoch 207: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1114 - mae: 0.3396 - val_loss: 0.0897 - val_mae: 0.2903 - learning_rate: 4.9000e-05
Epoch 208/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1084 - mae: 0.3322
Epoch 208: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1058 - mae: 0.3266 - val_loss: 0.0899 - val_mae: 0.2891 - learning_rate: 4.9000e-05
Epoch 209/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1116 - mae: 0.3391
Epoch 209: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1090 - mae: 0.3363 - val_loss: 0.1053 - val_mae: 0.3210 - learning_rate: 4.9000e-05
Epoch 210/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1038 - mae: 0.3229
Epoch 210: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1043 - mae: 0.3242 - val_loss: 0.0960 - val_mae: 0.3074 - learning_rate: 4.9000e-05
Epoch 211/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1072 - mae: 0.3309
Epoch 211: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1043 - mae: 0.3250 - val_loss: 0.1010 - val_mae: 0.3136 - learning_rate: 4.9000e-05
Epoch 212/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1043 - mae: 0.3236
Epoch 212: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1036 - mae: 0.3223 - val_loss: 0.1029 - val_mae: 0.3211 - learning_rate: 4.9000e-05
Epoch 213/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1027 - mae: 0.3212
Epoch 213: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1042 - mae: 0.3262 - val_loss: 0.0979 - val_mae: 0.3071 - learning_rate: 4.9000e-05
Epoch 214/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1079 - mae: 0.3340
Epoch 214: ReduceLROnPlateau reducing learning rate to 3.4299996332265434e-05.

Epoch 214: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1067 - mae: 0.3305 - val_loss: 0.0959 - val_mae: 0.3037 - learning_rate: 4.9000e-05
Epoch 215/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1013 - mae: 0.3209
Epoch 215: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1045 - mae: 0.3255 - val_loss: 0.0900 - val_mae: 0.2950 - learning_rate: 3.4300e-05
Epoch 216/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1035 - mae: 0.3220
Epoch 216: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1051 - mae: 0.3258 - val_loss: 0.0995 - val_mae: 0.3152 - learning_rate: 3.4300e-05
Epoch 217/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1071 - mae: 0.3297
Epoch 217: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1060 - mae: 0.3269 - val_loss: 0.0973 - val_mae: 0.3106 - learning_rate: 3.4300e-05
Epoch 218/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0981 - mae: 0.3158
Epoch 218: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1025 - mae: 0.3225 - val_loss: 0.0929 - val_mae: 0.2931 - learning_rate: 3.4300e-05
Epoch 219/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1098 - mae: 0.3371
Epoch 219: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1101 - mae: 0.3388 - val_loss: 0.0966 - val_mae: 0.3061 - learning_rate: 3.4300e-05
Epoch 220/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1065 - mae: 0.3282
Epoch 220: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1056 - mae: 0.3274 - val_loss: 0.0964 - val_mae: 0.3036 - learning_rate: 3.4300e-05
Epoch 221/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0995 - mae: 0.3136
Epoch 221: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0995 - mae: 0.3130 - val_loss: 0.0932 - val_mae: 0.3029 - learning_rate: 3.4300e-05
Epoch 222/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1064 - mae: 0.3320
Epoch 222: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1015 - mae: 0.3198 - val_loss: 0.0914 - val_mae: 0.2943 - learning_rate: 3.4300e-05
Epoch 223/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1044 - mae: 0.3221
Epoch 223: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1034 - mae: 0.3233 - val_loss: 0.0970 - val_mae: 0.3040 - learning_rate: 3.4300e-05
Epoch 224/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1018 - mae: 0.3189
Epoch 224: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1033 - mae: 0.3221 - val_loss: 0.0982 - val_mae: 0.3169 - learning_rate: 3.4300e-05
Epoch 225/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0998 - mae: 0.3173
Epoch 225: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1038 - mae: 0.3238 - val_loss: 0.0968 - val_mae: 0.3067 - learning_rate: 3.4300e-05
Epoch 226/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0988 - mae: 0.3148
Epoch 226: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1018 - mae: 0.3212 - val_loss: 0.1006 - val_mae: 0.3161 - learning_rate: 3.4300e-05
Epoch 227/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1033 - mae: 0.3227
Epoch 227: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1017 - mae: 0.3206 - val_loss: 0.0922 - val_mae: 0.2958 - learning_rate: 3.4300e-05
Epoch 228/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1034 - mae: 0.3238
Epoch 228: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1043 - mae: 0.3258 - val_loss: 0.0906 - val_mae: 0.2954 - learning_rate: 3.4300e-05
Epoch 229/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1059 - mae: 0.3299
Epoch 229: ReduceLROnPlateau reducing learning rate to 2.400999692326877e-05.

Epoch 229: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1064 - mae: 0.3297 - val_loss: 0.0910 - val_mae: 0.2957 - learning_rate: 3.4300e-05
Epoch 230/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1022 - mae: 0.3208
Epoch 230: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1013 - mae: 0.3210 - val_loss: 0.0917 - val_mae: 0.2963 - learning_rate: 2.4010e-05
Epoch 231/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0985 - mae: 0.3152
Epoch 231: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0992 - mae: 0.3159 - val_loss: 0.0959 - val_mae: 0.3048 - learning_rate: 2.4010e-05
Epoch 232/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1036 - mae: 0.3257
Epoch 232: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1012 - mae: 0.3220 - val_loss: 0.0907 - val_mae: 0.2933 - learning_rate: 2.4010e-05
Epoch 233/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1003 - mae: 0.3186
Epoch 233: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1018 - mae: 0.3235 - val_loss: 0.0928 - val_mae: 0.3005 - learning_rate: 2.4010e-05
Epoch 234/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0975 - mae: 0.3123
Epoch 234: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1010 - mae: 0.3205 - val_loss: 0.0948 - val_mae: 0.3043 - learning_rate: 2.4010e-05
Epoch 235/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0967 - mae: 0.3102
Epoch 235: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1009 - mae: 0.3194 - val_loss: 0.0933 - val_mae: 0.3013 - learning_rate: 2.4010e-05
Epoch 236/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1023 - mae: 0.3237
Epoch 236: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1054 - mae: 0.3301 - val_loss: 0.0955 - val_mae: 0.3049 - learning_rate: 2.4010e-05
Epoch 237/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1004 - mae: 0.3201
Epoch 237: val_loss did not improve from 0.08877
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1016 - mae: 0.3229 - val_loss: 0.1019 - val_mae: 0.3138 - learning_rate: 2.4010e-05
Epoch 238/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1074 - mae: 0.3311
Epoch 238: val_loss improved from 0.08877 to 0.08794, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 238: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1045 - mae: 0.3252 - val_loss: 0.0879 - val_mae: 0.2890 - learning_rate: 2.4010e-05
Epoch 239/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1018 - mae: 0.3226
Epoch 239: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1007 - mae: 0.3195 - val_loss: 0.0972 - val_mae: 0.3103 - learning_rate: 2.4010e-05
Epoch 240/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1075 - mae: 0.3324
Epoch 240: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1028 - mae: 0.3240 - val_loss: 0.0917 - val_mae: 0.2970 - learning_rate: 2.4010e-05
Epoch 241/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0991 - mae: 0.3131
Epoch 241: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0983 - mae: 0.3132 - val_loss: 0.0943 - val_mae: 0.3026 - learning_rate: 2.4010e-05
Epoch 242/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0993 - mae: 0.3192
Epoch 242: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1006 - mae: 0.3205 - val_loss: 0.0986 - val_mae: 0.3045 - learning_rate: 2.4010e-05
Epoch 243/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1005 - mae: 0.3201
Epoch 243: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1019 - mae: 0.3234 - val_loss: 0.0915 - val_mae: 0.2980 - learning_rate: 2.4010e-05
Epoch 244/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0987 - mae: 0.3168
Epoch 244: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0994 - mae: 0.3171 - val_loss: 0.0916 - val_mae: 0.2957 - learning_rate: 2.4010e-05
Epoch 245/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0950 - mae: 0.3077
Epoch 245: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0986 - mae: 0.3148 - val_loss: 0.0932 - val_mae: 0.2989 - learning_rate: 2.4010e-05
Epoch 246/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1021 - mae: 0.3230
Epoch 246: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1014 - mae: 0.3222 - val_loss: 0.0905 - val_mae: 0.2959 - learning_rate: 2.4010e-05
Epoch 247/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1021 - mae: 0.3219
Epoch 247: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1020 - mae: 0.3236 - val_loss: 0.0956 - val_mae: 0.3060 - learning_rate: 2.4010e-05
Epoch 248/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1068 - mae: 0.3327
Epoch 248: val_loss did not improve from 0.08794
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1042 - mae: 0.3272 - val_loss: 0.0919 - val_mae: 0.2962 - learning_rate: 2.4010e-05
Epoch 249/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0988 - mae: 0.3152
Epoch 249: val_loss improved from 0.08794 to 0.08730, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 249: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.0991 - mae: 0.3155 - val_loss: 0.0873 - val_mae: 0.2886 - learning_rate: 2.4010e-05
Epoch 250/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0981 - mae: 0.3133
Epoch 250: val_loss did not improve from 0.08730
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0994 - mae: 0.3167 - val_loss: 0.0964 - val_mae: 0.3106 - learning_rate: 2.4010e-05
Epoch 251/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1001 - mae: 0.3186
Epoch 251: val_loss did not improve from 0.08730
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1005 - mae: 0.3209 - val_loss: 0.1044 - val_mae: 0.3290 - learning_rate: 2.4010e-05
Epoch 252/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1031 - mae: 0.3276
Epoch 252: val_loss did not improve from 0.08730
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1048 - mae: 0.3285 - val_loss: 0.0906 - val_mae: 0.2946 - learning_rate: 2.4010e-05
Epoch 253/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0947 - mae: 0.3092
Epoch 253: val_loss did not improve from 0.08730
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0976 - mae: 0.3143 - val_loss: 0.1031 - val_mae: 0.3212 - learning_rate: 2.4010e-05
Epoch 254/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0982 - mae: 0.3192
Epoch 254: val_loss did not improve from 0.08730
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1016 - mae: 0.3251 - val_loss: 0.0924 - val_mae: 0.2965 - learning_rate: 2.4010e-05
Epoch 255/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0999 - mae: 0.3198
Epoch 255: val_loss improved from 0.08730 to 0.08603, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 255: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.0980 - mae: 0.3145 - val_loss: 0.0860 - val_mae: 0.2848 - learning_rate: 2.4010e-05
Epoch 256/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1002 - mae: 0.3225
Epoch 256: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0999 - mae: 0.3205 - val_loss: 0.0901 - val_mae: 0.2952 - learning_rate: 2.4010e-05
Epoch 257/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0992 - mae: 0.3169
Epoch 257: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1014 - mae: 0.3192 - val_loss: 0.1060 - val_mae: 0.3279 - learning_rate: 2.4010e-05
Epoch 258/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0959 - mae: 0.3084
Epoch 258: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0963 - mae: 0.3109 - val_loss: 0.0944 - val_mae: 0.3023 - learning_rate: 2.4010e-05
Epoch 259/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1020 - mae: 0.3209
Epoch 259: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1002 - mae: 0.3184 - val_loss: 0.0991 - val_mae: 0.3116 - learning_rate: 2.4010e-05
Epoch 260/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0969 - mae: 0.3139
Epoch 260: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0979 - mae: 0.3149 - val_loss: 0.0949 - val_mae: 0.3034 - learning_rate: 2.4010e-05
Epoch 261/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0974 - mae: 0.3131
Epoch 261: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0985 - mae: 0.3162 - val_loss: 0.0981 - val_mae: 0.3115 - learning_rate: 2.4010e-05
Epoch 262/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0955 - mae: 0.3129
Epoch 262: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0953 - mae: 0.3097 - val_loss: 0.0987 - val_mae: 0.3109 - learning_rate: 2.4010e-05
Epoch 263/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0964 - mae: 0.3115
Epoch 263: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0983 - mae: 0.3150 - val_loss: 0.0906 - val_mae: 0.2986 - learning_rate: 2.4010e-05
Epoch 264/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1018 - mae: 0.3256
Epoch 264: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1014 - mae: 0.3228 - val_loss: 0.0923 - val_mae: 0.2999 - learning_rate: 2.4010e-05
Epoch 265/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1036 - mae: 0.3271
Epoch 265: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1023 - mae: 0.3235 - val_loss: 0.0969 - val_mae: 0.3101 - learning_rate: 2.4010e-05
Epoch 266/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1021 - mae: 0.3253
Epoch 266: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1048 - mae: 0.3283 - val_loss: 0.0917 - val_mae: 0.2967 - learning_rate: 2.4010e-05
Epoch 267/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1001 - mae: 0.3228
Epoch 267: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0991 - mae: 0.3169 - val_loss: 0.0910 - val_mae: 0.2963 - learning_rate: 2.4010e-05
Epoch 268/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0970 - mae: 0.3154
Epoch 268: val_loss did not improve from 0.08603
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1003 - mae: 0.3219 - val_loss: 0.0979 - val_mae: 0.3107 - learning_rate: 2.4010e-05
Epoch 269/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1059 - mae: 0.3303
Epoch 269: val_loss improved from 0.08603 to 0.08481, saving model to /content/drive/MyDrive/cache/best_model.keras

Epoch 269: finished saving model to /content/drive/MyDrive/cache/best_model.keras
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - loss: 0.1020 - mae: 0.3231 - val_loss: 0.0848 - val_mae: 0.2847 - learning_rate: 2.4010e-05
Epoch 270/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0967 - mae: 0.3135
Epoch 270: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0985 - mae: 0.3180 - val_loss: 0.0919 - val_mae: 0.2987 - learning_rate: 2.4010e-05
Epoch 271/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0984 - mae: 0.3159
Epoch 271: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0979 - mae: 0.3150 - val_loss: 0.0881 - val_mae: 0.2911 - learning_rate: 2.4010e-05
Epoch 272/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0971 - mae: 0.3143
Epoch 272: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0975 - mae: 0.3159 - val_loss: 0.0979 - val_mae: 0.3087 - learning_rate: 2.4010e-05
Epoch 273/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0969 - mae: 0.3138
Epoch 273: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1011 - mae: 0.3230 - val_loss: 0.0895 - val_mae: 0.2948 - learning_rate: 2.4010e-05
Epoch 274/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1006 - mae: 0.3220
Epoch 274: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1014 - mae: 0.3224 - val_loss: 0.0914 - val_mae: 0.2970 - learning_rate: 2.4010e-05
Epoch 275/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0963 - mae: 0.3118
Epoch 275: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0958 - mae: 0.3116 - val_loss: 0.0928 - val_mae: 0.3021 - learning_rate: 2.4010e-05
Epoch 276/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1010 - mae: 0.3187
Epoch 276: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0987 - mae: 0.3148 - val_loss: 0.0931 - val_mae: 0.3016 - learning_rate: 2.4010e-05
Epoch 277/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1017 - mae: 0.3229
Epoch 277: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.1001 - mae: 0.3202 - val_loss: 0.0956 - val_mae: 0.3071 - learning_rate: 2.4010e-05
Epoch 278/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.1069 - mae: 0.3302
Epoch 278: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0995 - mae: 0.3167 - val_loss: 0.0889 - val_mae: 0.2921 - learning_rate: 2.4010e-05
Epoch 279/1000
312/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0959 - mae: 0.3134
Epoch 279: val_loss did not improve from 0.08481
313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 79ms/step - loss: 0.0979 - mae: 0.3173 - val_loss: 0.0925 - val_mae: 0.3000 - learning_rate: 2.4010e-05
Epoch 280/1000
302/313 ━━━━━━━━━━━━━━━━━━━━ 0s 77ms/step - loss: 0.0979 - mae: 0.3196

"""

# =========================
# 📌 REGEX PARSE
# =========================
epoch_list = []
loss_list = []
mae_list = []
val_loss_list = []
val_mae_list = []

pattern = re.compile(
    r"Epoch\s+(\d+)/\d+.*?loss:\s*([0-9\.]+).*?mae:\s*([0-9\.]+).*?val_loss:\s*([0-9\.]+).*?val_mae:\s*([0-9\.]+)",
    re.S
)

matches = pattern.findall(log_text)

for m in matches:
    epoch_list.append(int(m[0]))
    loss_list.append(float(m[1]))
    mae_list.append(float(m[2]))
    val_loss_list.append(float(m[3]))
    val_mae_list.append(float(m[4]))

print(f"Parsed {len(epoch_list)} epochs")

# =========================
# 📌 PLOT LOSS
# =========================
plt.figure()
plt.plot(epoch_list, loss_list, label="train_loss")
plt.plot(epoch_list, val_loss_list, label="val_loss")


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid()
plt.show()

# =========================
# 📌 PLOT MAE
# =========================
plt.figure()
plt.plot(epoch_list, mae_list, label="train_mae")
plt.plot(epoch_list, val_mae_list, label="val_mae")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE vs Epoch")
plt.legend()
plt.grid()
plt.show()