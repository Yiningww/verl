import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})

# ---------- 可调参数 ----------
OUTPUT = "3b-gsm8k.pdf"
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
        'token_mean':[72.9, 82.9, 84.1, 84.3, 85.1],
        'seq_mean_token_mean':[72.9, 83.6, 83.5, 84.5, 85.7],
         'ours':[72.9, 84.3, 84.5, 85.3, 85.3],
         'cutoff':82,
         'yticks':[0,82,85,88],
        }

math500 = {'title': 'Math500',
        'token_mean':[50.8, 63.8, 65.8, 65.6, 65.4],
        'seq_mean_token_mean':[50.8, 64.6, 64.8, 65.4, 64],
         'ours':[50.8, 65.4, 66.4, 67.4, 66.4],
         'cutoff':62,
         'yticks':[0,62, 65, 68],
        }

minerva = {'title': 'Minerva',
        'token_mean':[17, 31.2, 29, 30.5, 31.2],
        'seq_mean_token_mean':[17, 27.9, 29.4, 30.5, 31.2],
         'ours':[17, 32, 34.2, 32.7, 32.4],
         'cutoff':28,
         'yticks':[0, 28, 31, 33],
        }

gaokao = {'title': 'Gaokao',
        'token_mean':[48.3, 52.8, 52.3, 50.6, 53],
        'seq_mean_token_mean':[48.3, 53.8, 52.8, 52.3, 53],
         'ours':[48.3, 54, 53.7, 54, 53.9],
         'cutoff':52,
         'yticks':[0,52, 55, 58],
        }

olympiad = {'title': 'Olympiad',
        'token_mean':[17, 27.1, 26.5, 27.3, 27.6],
        'seq_mean_token_mean':[17, 23.7, 27, 28, 27.9],
         'ours':[17, 27.1, 28.1, 27.7, 28.1],
         'cutoff':25,
         'yticks':[0, 25, 27, 29],
        }

college_math = {'title': 'College Math',
        'token_mean':[35.5, 35.3, 36.6, 36.2, 36.6],
        'seq_mean_token_mean':[35.5, 35.8, 35.6, 36.4, 37],
         'ours':[35.5, 36.8, 38.2, 37.2, 37.6],
         'cutoff':36,
         'yticks':[0, 36, 38, 40],
        }

aime_avg_32 = {'title': 'AIME avg@32',
        'token_mean':[0, 9.4, 6.2, 7.7, 6.2],
        'seq_mean_token_mean':[0, 7.9, 10.2, 7, 7.7],
         'ours':[0, 10.6, 7.3, 7.3, 8.1],
         'cutoff':7.0,
         'yticks':[0, 7.0, 9.0, 11.0],
        }

amc_avg_32 = {'title': 'AMC avg@32',
        'token_mean':[0.3, 38.4, 38.6, 34.1, 35.9],
        'seq_mean_token_mean':[0.3, 38.9, 40.2, 38.8, 40.8],
         'ours':[0.3, 40.8, 38.4, 38.7, 38.5],
         'cutoff':34,
         'yticks':[0, 34, 38, 42],
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
    ax.plot(steps_s, ours_s, 'o-', color="tab:red", markersize=4, label="ours")
    ax.plot(steps_s, tm_s,   'o-', color="tab:blue", markersize=4, label="token-mean")
    ax.plot(steps_s, seq_s,  'o-', color="tab:green", markersize=4, label="seq-mean-token-mean")

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