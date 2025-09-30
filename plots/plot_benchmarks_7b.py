import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})

# ---------- 可调参数 ----------
OUTPUT = "7b-benchmarks.pdf"
FIGSIZE = (3, 2.5)
DPI = 220
SMOOTH_WINDOW = 1
STEPS = np.array([0, 40, 80, 120, 160], dtype=float)  # <<< 用真实 step
XTICKS = [0, 40, 80, 120, 160]
XMIN, XMAX = 0, 160

def moving_average(x, w):
    x = np.asarray(x, float)
    if w <= 1:
        return x
    if w % 2 == 0:
        w += 1
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    ker = np.ones(w)/w
    return np.convolve(xp, ker, mode="valid")

def make_func(cutoff):
    def forward(y):
        if y <= cutoff:
            return (y / cutoff) * 0.5      # 压缩到 [0,0.5]
        else:
            return 0.5 + (y - cutoff) / 10 * 1.5   # 拉伸到 [0.5,2]

    def inverse(y_new):
        if y_new <= 0.5:
            return y_new / 0.5 * cutoff
        else:
            return cutoff + (y_new - 0.5) / 1.5 * 10
    return forward, inverse

# ---------- 数据 ----------

GSM8k = {'title': 'GSM8k',
        'token_mean':[87.8, 90.8, 90.8, 91, 91.4],
        'seq_mean_token_mean':[87.8, 89.7, 90.8, 90.7, 91.5],
         'ours':[87.8, 90.5, 90.8, 92, 91.8],
         'cutoff':89,
         'yticks':[0, 89, 91, 93],
        }

math500 = {'title': 'Math500',
        'token_mean':[63.6, 75.2, 75, 77.8, 75.8],
        'seq_mean_token_mean':[63.6, 77.2, 75.6, 75.8, 75.2],
         'ours':[63.6, 75.8, 76, 76.1, 76.3],
         'cutoff':75,
         'yticks':[0, 75, 77, 79],
        }

minerva = {'title': 'Minerva',
        'token_mean':[26.8, 39.3, 37.1, 36.4, 36.8],
        'seq_mean_token_mean':[26.8, 40.2, 39.4, 35.7, 35.7],
         'ours':[26.8, 39.6, 39.9, 40.1, 39.3],
         'cutoff':35,
         'yticks':[0, 35, 38, 41],
        }

gaokao = {'title': 'Gaokao',
        'token_mean':[53.5, 63.4, 63.4, 63.9, 64.4],
        'seq_mean_token_mean':[53.5, 62.6, 60.8, 60.5, 60.8],
         'ours':[53.5, 63.1, 63.8, 66.2, 65.1],
         'cutoff':61,
         'yticks':[0, 61, 63, 65],
        }

olympiad = {'title': 'Olympiad',
        'token_mean':[29.9, 36.7, 38.5, 37.9, 39.4],
        'seq_mean_token_mean':[29.9, 35.6, 36.9, 37.9, 35.7],
         'ours':[29.9, 37, 37.3, 38.1, 38.7],
         'cutoff':35,
         'yticks':[0, 35, 38, 41],
        }

college_math = {'title': 'College Math',
        'token_mean':[34.6, 40.4, 39.8, 40.4, 40.2],
        'seq_mean_token_mean':[34.6, 39.6, 40, 41.2, 39.8],
         'ours':[34.6, 40.2, 41.4, 41.4, 40.9],
         'cutoff':39,
         'yticks':[0, 39, 41, 43],
        }

aime_avg_32 = {'title': 'AIME avg@32',
        'token_mean':[8, 14, 16.2, 14.2, 15.2],
        'seq_mean_token_mean':[8, 17.2, 17.1, 11.4, 16.1],
         'ours':[8, 14.7, 14.3, 15.1, 14.7],
         'cutoff':11.0,
         'yticks':[0, 11.0, 14.0, 17.0],
        }

amc_avg_32 = {'title': 'AMC avg@32',
        'token_mean':[36.6, 51.1, 58.2, 54.1, 48.7],
        'seq_mean_token_mean':[36.6, 52.6, 49.5, 54.5, 51.6],
         'ours':[36.6, 51, 57.1, 58.8, 56.5],
         'cutoff':50,
         'yticks':[0, 50, 54, 58],
        }



data_list = [GSM8k,
             math500,
             minerva,
             gaokao,
             olympiad,
             college_math,
             aime_avg_32,
             amc_avg_32]


# ---------- matplotlib 默认设置 ----------
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})



# ---------- 画图 ----------
fig,axs = plt.subplots(2,4, figsize=(8,4),sharex=False,sharey=False)
fig.subplots_adjust(wspace=0.0)
fig.subplots_adjust(hspace=0.02)

for i, data in enumerate(data_list):
    
    # ---------- 读取数据 ----------
    
    title               = data['title']
    token_mean          = data['token_mean']
    seq_mean_token_mean = data['seq_mean_token_mean']
    ours                = data['ours']
    
    # ---------- 对齐长度 ----------
    N = min(len(ours), len(token_mean), len(seq_mean_token_mean), len(STEPS))
    steps = STEPS[:N]
    ours_s = moving_average(ours[:N], SMOOTH_WINDOW)
    tm_s   = moving_average(token_mean[:N], SMOOTH_WINDOW)
    seq_s  = moving_average(seq_mean_token_mean[:N], SMOOTH_WINDOW)
    steps_s = steps[:len(ours_s)]  # 与平滑后的长度对齐
    
    ax = axs.flatten()[i]
    ax.plot(steps_s, ours_s, 'o-', color="#b63120", markersize=4, label=r'$\lambda$-GRPO')
    ax.plot(steps_s, tm_s,   'o-', color="#fb8122", markersize=4, label="DAPO")
    ax.plot(steps_s, seq_s,  'o-', color="#e8d174", markersize=4, label="GRPO")

    # 设置自定义 y 轴
    forward, inverse = make_func(data['cutoff'])        
    fwd = np.vectorize(forward, otypes=[float])
    inv = np.vectorize(inverse, otypes=[float])
    ax.set_yscale('function', functions=(fwd, inv))

    # 自定义 y 轴刻度
    yticks = data['yticks']
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks])
    ax.set_title(title, fontweight="bold")
    
    # x axis
    ax.set_xlim(0, 165)
    # ax.set_ylim(60, 90)
    ax.set_xticks([t for t in XTICKS if XMIN <= t <= XMAX])

    # 在 y=70 加一条虚线，提示“分段” / or add grid
    #ax.axhline(70, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.grid(ls='--')



fig.text(-0.01, 0.5, "Accuracy %", fontweight="bold", va="center", rotation="vertical", fontsize=12)
fig.text(0.505, -0.01, "Step", fontweight="bold", ha="center", fontsize=12)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", ncol=3,
           bbox_to_anchor=(0.51, 0.95),fontsize=12)


    
plt.tight_layout()
# plt.show()
plt.savefig(OUTPUT, dpi=DPI, bbox_inches="tight")
print(f"Saved: {OUTPUT}")