import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})

# ---------- 可调参数 ----------
OUTPUT = "1.5b-gsm8k.pdf"
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
        'token_mean':[8.5, 72.9, 74.5, 75.7, 76.0],
        'seq_mean_token_mean':[8.5, 72.9, 72.9, 75.1, 75.1],
         'ours':[8.5, 73.4, 76.1, 77.3, 76.6],
         'cutoff':70,
         'yticks':[0,70,75,80],
        }

math500 = {'title': 'Math500',
        'token_mean':[1.8, 56.6, 58, 58, 56.4],
        'seq_mean_token_mean':[1.8, 55.2, 57, 56.8, 57],
         'ours':[1.8, 59.2, 58, 59.2, 58],
         'cutoff':55,
         'yticks':[0,55,58,61],
        }

minerva = {'title': 'Minerva',
        'token_mean':[1.8, 17.2, 19, 20.2, 18.8],
        'seq_mean_token_mean':[1.8, 17.6, 18, 21, 21.7],
         'ours':[1.8, 18.6, 19.9, 21, 20.9],
         'cutoff':15,
         'yticks':[0,15,20,25],
        }

gaokao = {'title': 'Gaokao',
        'token_mean':[1.8, 44.7, 47.4, 50.4, 47.8],
        'seq_mean_token_mean':[1.8, 45.2, 45.7, 48.6, 49.6],
         'ours':[1.8, 46.5, 49.8, 50.6, 50.4],
         'cutoff':43,
         'yticks':[0,45,50,55],
        }

olympiad = {'title': 'Olympiad',
        'token_mean':[1.3, 18.4, 19.9, 20.6, 19.8],
        'seq_mean_token_mean':[1.3, 16.7, 18.2, 19.9, 20.3],
         'ours':[1.3, 19.5, 20.4, 21.3, 21],
         'cutoff':15,
         'yticks':[0,15,20,25],
        }

college_math = {'title': 'College Math',
        'token_mean':[0, 32.8, 32.4, 33, 31.6],
        'seq_mean_token_mean':[0, 32.7, 33.5, 32.1, 32.4],
         'ours':[0, 33.6, 35.2, 35, 34.2],
         'cutoff':30,
         'yticks':[0,30,35,40],
        }

aime_avg_32 = {'title': 'AIME avg@32',
        'token_mean':[0, 1.6, 2.5, 3.3, 0.7],
        'seq_mean_token_mean':[0, 2.8, 1.6, 2.5, 4.2],
         'ours':[0, 1.6, 3.1, 3.4, 4.6],
         'cutoff':1.5,
         'yticks':[0,1.5,3.5,5.5],
        }

amc_avg_32 = {'title': 'AMC avg@32',
        'token_mean':[0.3, 25.6, 31.6, 31.5, 29.1],
        'seq_mean_token_mean':[0.3, 25.5, 24.9, 25.5, 26.9],
         'ours':[0.3, 24.7, 27.4, 34.7, 37.7],
         'cutoff':20,
         'yticks':[0,20,30,40],
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
