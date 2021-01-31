import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

from utils.data_handling import load_config


def get_colors():
    cfg = load_config()
    sb.set_style(cfg['plotting']['seaborn']['style'])
    sb.set_context(cfg['plotting']['seaborn']['context']['context'],
                   font_scale=cfg['plotting']['seaborn']['context']['font scale'],
                   rc=cfg['plotting']['seaborn']['context']['rc'])
    colors = list(sb.color_palette(cfg['plotting']['palette']))
    color_map = cfg['plotting']['color_map']
    return colors, color_map


def plot_lens(ax, methods, lens, steps_per_ep, title, episodes=50_000,
              log=False, logx=False, eval_steps=1):
    colors, color_map = get_colors()
    print('AVG pol length')
    mv = 0
    created_steps_leg = False
    for method in methods:
        if method != 'hq':
            m, s = lens[method].mean(axis=0), lens[method].std(axis=0)
            total_ones = np.ones(m.shape) * 100
            # print("{:>4s}: {:>5.2f}".format(method, 1 - np.sum(total_ones - m) / np.sum(total_ones)))
            print("{:>4s}: {:>5.2f}".format(method, np.mean(m)))
            ax.step(np.arange(1, m.shape[0] + 1) * eval_steps, m, where='post',
                    c=colors[color_map[method]])
            ax.fill_between(
                np.arange(1, m.shape[0] + 1) * eval_steps, m - s, m + s, alpha=0.25, step='post',
                color=colors[color_map[method]])
            if 'hq' not in methods:
                mv = max(mv, max(m) + max(m) * .05)
        else:
            raise NotImplementedError
        if steps_per_ep:
            m, s = steps_per_ep[method].mean(axis=0), steps_per_ep[method].std(axis=0)
            ax.step(np.arange(1, m.shape[0] + 1) * eval_steps, m, where='post',
                    c=np.array(colors[color_map[method]]) * .75, ls=':')
            ax.fill_between(
                np.arange(1, m.shape[0] + 1) * eval_steps, m - s, m + s, alpha=0.125, step='post',
                color=np.array(colors[color_map[method]]) * .75)
            mv = max(mv, max(m))
            if not created_steps_leg:
                ax.plot([-999, -999], [-999, -999], ls=':', c='k', label='all')
                ax.plot([-999, -999], [-999, -999], ls='-', c='k', label='dec')
                created_steps_leg = True
    if log:
        ax.set_ylim([1, max(mv, 100)])
        ax.semilogy()
    else:
        ax.set_ylim([0, mv])
    if logx:
        ax.set_ylim([1, max(mv, 100)])
        ax.set_xlim([1, episodes * eval_steps])
        ax.semilogx()
    else:
        ax.set_xlim([0, episodes * eval_steps])
        ax.set_ylabel('#Steps')
    if steps_per_ep:
        ax.legend(loc='upper right', ncol=1, handlelength=.75)
    ax.set_ylabel('#Steps')
    ax.set_xlabel('#Episodes')
    ax.set_title(title)
    return ax


def _annotate(ax, rewards, max_reward, eval_steps):
    qxy = ((np.where(rewards['q'].mean(axis=0) >= .5 * max_reward)[0])[0] * eval_steps, .5)
    sqvxy = ((np.where(rewards['sq'].mean(axis=0) >= .5 * max_reward)[0])[0] * eval_steps, .5)
    ax.annotate("",  # '{:d}x speedup'.format(int(np.round(qxy[0]/sqvxy[0]))),
                xy=qxy,
                xycoords='data', xytext=sqvxy, textcoords='data',
                arrowprops=dict(arrowstyle="<->", color="0.",
                                connectionstyle="arc3,rad=0.", lw=5,
                                ), )

    speedup = qxy[0] / sqvxy[0]
    qxy = (qxy[0], .5 * max_reward)
    sqvxy = (sqvxy[0], .25)
    ax.annotate(r'${:.2f}\times$ speedup'.format(speedup),
                xy=qxy,
                xycoords='data', xytext=sqvxy, textcoords='data',
                arrowprops=dict(arrowstyle="-", color="0.",
                                connectionstyle="arc3,rad=0.", lw=0
                                ),
                fontsize=22)

    try:
        qxy = ((np.where(rewards['q'].mean(axis=0) >= max_reward)[0])[0] * eval_steps, max_reward)
        sqvxy = ((np.where(rewards['sq'].mean(axis=0) >= max_reward)[0])[0] * eval_steps, max_reward)
        ax.annotate("",  # '{:d}x speedup'.format(int(np.round(qxy[0]/sqvxy[0]))),
                    xy=qxy,
                    xycoords='data', xytext=sqvxy, textcoords='data',
                    arrowprops=dict(arrowstyle="<->", color="0.",
                                    connectionstyle="arc3,rad=0.", lw=5,
                                    ), )

        speedup = qxy[0] / sqvxy[0]
        qxy = (qxy[0], max_reward)
        sqvxy = (sqvxy[0], .75)
        ax.annotate(r'${:.2f}\times$ speedup'.format(speedup),
                    xy=qxy,
                    xycoords='data', xytext=sqvxy, textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0.",
                                    connectionstyle="arc3,rad=0.", lw=0
                                    ),
                    fontsize=22)
    except:
        pass


def plot_rewards(ax, methods, rewards, title, episodes=50_000,
                 xlabel='#Episodes', log=False, logx=False, annotate=False, eval_steps=1):
    colors, color_map = get_colors()
    print('AUC')
    min_m = np.inf
    max_m = -np.inf
    for method in methods:
        m, s = rewards[method].mean(axis=0), rewards[method].std(axis=0)
        # used for AUC computation
        m_, s_ = ((rewards[method] + 1) / 2).mean(axis=0), ((rewards[method] + 1) / 2).std(axis=0)
        min_m = min(min(m), min_m)
        max_m = max(max(m), max_m)
        total_ones = np.ones(m.shape)
        label = method
        if method == 'sqv3':
            label = "sn-$\mathcal{Q}$"
        elif method == 'sq':
            label = "t-$\mathcal{Q}$"
        label = label.replace('q', '$\mathcal{Q}$')
        label = r'{:s}'.format(label)
        print("{:>2s}: {:>5.2f}".format(method, 1 - np.sum(total_ones - m_) / np.sum(total_ones)))
        ax.step(np.arange(1, m.shape[0] + 1) * eval_steps, m, where='post', c=colors[color_map[method]],
                label=label)
        ax.fill_between(np.arange(1, m.shape[0] + 1) * eval_steps, m - s, m + s, alpha=0.25, step='post',
                        color=colors[color_map[method]])
    if annotate:
        _annotate(ax, rewards, max_m, eval_steps)
    if log:
        ax.set_ylim([.01, max_m])
        ax.semilogy()
    else:
        ax.set_ylim([min(-1, min_m - .1), max(1, max_m + .1)])
    if logx:
        ax.set_xlim([1, episodes * eval_steps])
        ax.semilogx()
    else:
        ax.set_xlim([0, episodes * eval_steps])
    ax.set_ylabel('Reward')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(ncol=1, loc='lower right', handlelength=.75)
    return ax


def plot(methods, rewards, lens, steps_per_ep, title, episodes=50_000,
         show=True, savefig=None, logleny=True,
         logrewy=True, logx=False, annotate=False, eval_steps=1, horizontal=False,
         individual=False):
    _, _ = get_colors()
    if not individual:
        if horizontal:
            fig, ax = plt.subplots(1, 2, figsize=(32, 5), dpi=100)
        else:
            fig, ax = plt.subplots(2, figsize=(20, 10), dpi=100)
        ax[0] = plot_rewards(ax[0], methods, rewards, title, episodes,
                             xlabel='#Episodes' if horizontal else '',
                             log=logrewy, logx=logx, annotate=annotate, eval_steps=eval_steps)
        print()
        ax[1] = plot_lens(ax[1], methods, lens, steps_per_ep, '', episodes, log=logleny,
                          logx=logx, eval_steps=eval_steps)
        if savefig:
            plt.savefig(savefig, dpi=100)
        if show:
            plt.show()
    else:
        try:
            name, suffix = savefig.split('.')
        except:
            name = savefig
            suffix = '.pdf'
        fig, ax = plt.subplots(1, figsize=(10, 4), dpi=100)
        ax = plot_rewards(ax, methods, rewards, '', episodes,
                          xlabel='#Episodes',
                          log=logrewy, logx=logx, annotate=annotate, eval_steps=eval_steps)
        plt.tight_layout()
        if savefig:
            plt.savefig(name + '_rewards' + '.' + suffix, dpi=100)
        if show:
            plt.show()
        fig, ax = plt.subplots(1, figsize=(10, 4), dpi=100)
        ax = plot_lens(ax, methods, lens, steps_per_ep, '', episodes, log=logleny,
                       logx=logx, eval_steps=eval_steps)
        plt.tight_layout()
        if savefig:
            plt.savefig(name + '_lens' + '.' + suffix, dpi=100)
        if show:
            plt.show()
