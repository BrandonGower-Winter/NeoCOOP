import animatplot as amp
import matplotlib.pyplot as plt
import numpy as np


def generateAnimat(title: str, records: [[float]], fps: int = 1, vmin=0, vmax=255, filename: str = 'animat',
                   x_label: str = 'X', y_label: str = 'Y'):

    fig, ax = plt.subplots()

    def animate(i):

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Generate the pix map
        plot = ax.imshow(records[i], interpolation='none', cmap='jet', vmin=vmin, vmax=vmax)

    blocks = amp.blocks.Nuke(animate, length=len(records), ax=ax)  # Required call to build our animation
    timeline = np.arange(len(records))
    anim = amp.Animation([blocks], amp.Timeline(timeline, fps=fps))  # Builds the Animation
    anim.controls()
    anim.save_gif(filename)


def generateBarAnimat(title: str, records: [[float]], bins: [], fps: int = 1, filename: str = 'animat',
                   x_label: str = 'X', y_label: str = 'Y', colors=None, labels=None):

    fig, ax = plt.subplots()

    def animate(i):

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        axis = np.arange(len(bins))
        ax.set_xticklabels(bins)

        width = 0.35

        for j in range(len(records[i])):
            ax.bar(axis + ((2*j - 1) * width), records[i][j],
                   color=colors[j] if colors is not None else None,
                   label=labels[j] if labels is not None else None)

        ax.legend(loc='center right')

    blocks = amp.blocks.Nuke(animate, length=len(records), ax=ax)  # Required call to build our animation
    timeline = np.arange(len(records))
    anim = amp.Animation([blocks], amp.Timeline(timeline, fps=fps))  # Builds the Animation
    anim.controls()
    anim.save_gif(filename)