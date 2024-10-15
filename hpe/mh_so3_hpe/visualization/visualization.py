from __future__ import print_function, absolute_import, division

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import seaborn as sns

# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp

from .utils import TEXT_WIDTH, setup_style

matplotlib.use("Agg")


def get_resolution(filename):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        filename,
    ]

    try:
        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
        for line in pipe.stdout:
            w, h = line.decode().strip().split(",")
    finally:
        pipe.stdout.close()

    return int(w), int(h)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)

    command = [
        "ffmpeg",
        "-i",
        filename,
        "-f",
        "image2pipe",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "-vcodec",
        "rawvideo",
        "-",
    ]

    i = 0
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    try:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > skip:
                yield np.frombuffer(data, dtype="uint8").reshape((h, w, 3))
            if i == limit:
                break
    finally:
        pipe.stdout.close()


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(
    keypoints,
    poses,
    skeleton,
    fps,
    bitrate,
    azim,
    output,
    viewport,
    limit=-1,
    downsample=1,
    size=6,
    input_video_path="",
    input_video_skip=0,
    elev=15,
):
    """Creates GIF or MP4 video of skeleton movements in 2D and 3D (in global
    frame)
    """

    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title("Input")

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        if len(data.shape) == 3:
            trajectories.append(data[:, 0, [0, 1]])
        else:
            trajectories.append(data[:, 0, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path == "":
        # Black background
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0]), dtype="uint8"
        )
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(
            np.array(all_frames), downsample
        ).astype("uint8")
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(
                trajectories[idx], downsample
            )
        fps /= downsample

    render_animation.initialized = False
    render_animation.image = None
    render_animation.lines = []
    render_animation.points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents

    def update_video(i):
        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d(
                [
                    -radius / 2 + trajectories[n][i, 0],
                    radius / 2 + trajectories[n][i, 0],
                ]
            )
            ax.set_ylim3d(
                [
                    -radius / 2 + trajectories[n][i, 1],
                    radius / 2 + trajectories[n][i, 1],
                ]
            )

        # Update 2D poses
        if not render_animation.initialized:
            render_animation.image = ax_in.imshow(
                all_frames[i], aspect="equal"
            )

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    # Draw skeleton only if keypoints match
                    # (otherwise we don't have the parents definition)
                    render_animation.lines.append(
                        ax_in.plot(
                            [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                            color="b",
                        )
                    )

                col = "red" if j in skeleton.joints_right else "black"
                cpal = sns.color_palette("Dark2")
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if len(pos.shape) == 2:
                        lines_3d[n].append(
                            ax.plot(
                                [pos[j, 0], pos[j_parent, 0]],
                                [pos[j, 1], pos[j_parent, 1]],
                                [pos[j, 2], pos[j_parent, 2]],
                                zdir="z",
                                c=col,
                            )
                        )
                    else:
                        for hyp, col in zip(pos, cpal):
                            lines_3d[n].append(
                                ax.plot(
                                    [hyp[j, 0], hyp[j_parent, 0]],
                                    [hyp[j, 1], hyp[j_parent, 1]],
                                    [hyp[j, 2], hyp[j_parent, 2]],
                                    zdir="z",
                                    c=col,
                                    alpha=hyp[j, 3] * 0.5 + 0.5 if hyp[j, 3] > 0.01 else 0.,  # <-- hyp score
                                    label=f"{hyp[j, 3]:.2f}" if j == 1 else None
                                )
                            )
                        ax.legend(loc="lower center", ncol=3)

            render_animation.points = ax_in.scatter(
                keypoints[i].T[0],
                keypoints[i].T[1],
                5,
                color="red",
                edgecolors="white",
                zorder=10,
            )

            render_animation.initialized = True
        else:
            render_animation.image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    render_animation.lines[j - 1][0].set_data(
                        [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                        [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                    )

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if len(pos.shape) == 2:
                        lines_3d[n][j - 1][0].set_xdata(
                            [pos[j, 0], pos[j_parent, 0]]
                        )
                        lines_3d[n][j - 1][0].set_ydata(
                            [pos[j, 1], pos[j_parent, 1]]
                        )
                        lines_3d[n][j - 1][0].set_3d_properties(
                            [pos[j, 2], pos[j_parent, 2]], zdir="z"
                        )
                    else:
                        n_hyp = len(pos)
                        for hyp_idx, hyp in enumerate(pos):
                            multi_hyp_j_idx = (j - 1) * n_hyp + hyp_idx
                            lines_3d[n][multi_hyp_j_idx][0].set_xdata(
                                [hyp[j, 0], hyp[j_parent, 0]]
                            )
                            lines_3d[n][multi_hyp_j_idx][0].set_ydata(
                                [hyp[j, 1], hyp[j_parent, 1]]
                            )
                            lines_3d[n][multi_hyp_j_idx][0].set_3d_properties(
                                [hyp[j, 2], hyp[j_parent, 2]], zdir="z"
                            )
                            lines_3d[n][multi_hyp_j_idx][0].set_alpha(
                                hyp[j, 3] * 0.5 + 0.5 if hyp[j, 3] > 0.01 else 0.
                            )
                            if j == 1:
                                lines_3d[n][multi_hyp_j_idx][0].set_label(
                                    f"{hyp[j, 3]:.2f}"
                                )
                            ax.legend(loc="lower center", ncol=3)

            render_animation.points.set_offsets(keypoints[i])

        print("{}/{}      ".format(i, limit), end="\r")

    fig.tight_layout()

    anim = FuncAnimation(
        fig,
        update_video,
        frames=np.arange(0, limit),
        interval=1000 / fps,
        repeat=False,
    )
    if output.endswith(".mp4"):
        Writer = writers["ffmpeg"]
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith(".gif"):
        anim.save(output, dpi=80, writer="imagemagick")
    else:
        raise ValueError(
            "Unsupported output format (only .mp4 and .gif are supported)"
        )
    plt.close()


def render_frame_prediction(
    frame_index,
    keypoints,
    poses,
    skeleton,
    azim,
    output,
    viewport,
    size=6,
    input_video_path="",
    input_video_skip=0,
    elev=15,
):
    """Creates pdf or png of predicted skeleton in 2D and 3D (in global
    frame)
    """

    plt.ioff()
    setup_style(fontsize=28, lw=2.0)
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title("Input")

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        if len(data.shape) == 3:
            trajectories.append(data[:, 0, [0, 1]])
        else:
            trajectories.append(data[:, 0, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path == "":
        # Black background
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0]), dtype="uint8"
        )
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    render_animation.initialized = False
    render_animation.image = None
    render_animation.lines = []
    render_animation.points = None

    parents = skeleton.parents

    def update_video(i):
        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d(
                [
                    -radius / 2 + trajectories[n][i, 0],
                    radius / 2 + trajectories[n][i, 0],
                ]
            )
            ax.set_ylim3d(
                [
                    -radius / 2 + trajectories[n][i, 1],
                    radius / 2 + trajectories[n][i, 1],
                ]
            )

        # Update 2D poses
        if not render_animation.initialized:
            render_animation.image = ax_in.imshow(
                all_frames[i], aspect="equal"
            )

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    # Draw skeleton only if keypoints match
                    # (otherwise we don't have the parents definition)
                    render_animation.lines.append(
                        ax_in.plot(
                            [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                            color="b",
                        )
                    )

                col = "red" if j in skeleton.joints_right else "black"
                cpal = sns.color_palette("Dark2")
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if len(pos.shape) == 2:
                        lines_3d[n].append(
                            ax.plot(
                                [pos[j, 0], pos[j_parent, 0]],
                                [pos[j, 1], pos[j_parent, 1]],
                                [pos[j, 2], pos[j_parent, 2]],
                                zdir="z",
                                c=col,
                            )
                        )
                    else:
                        for hyp, col in zip(pos, cpal):
                            lines_3d[n].append(
                                ax.plot(
                                    [hyp[j, 0], hyp[j_parent, 0]],
                                    [hyp[j, 1], hyp[j_parent, 1]],
                                    [hyp[j, 2], hyp[j_parent, 2]],
                                    zdir="z",
                                    c=col,
                                    alpha=hyp[j, 3] * 0.5 + 0.5 if hyp[j, 3] > 0.01 else 0.,  # <-- hyp score
                                    # label=f"{hyp[j, 3]:.2f}" if j == 1 else None
                                )
                            )
                        # ax.legend(loc="lower center", ncol=3)

            render_animation.points = ax_in.scatter(
                keypoints[i].T[0],
                keypoints[i].T[1],
                5,
                color="red",
                edgecolors="white",
                zorder=10,
            )

            render_animation.initialized = True
        else:
            render_animation.image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    render_animation.lines[j - 1][0].set_data(
                        [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                        [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                    )

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    if len(pos.shape) == 2:
                        lines_3d[n][j - 1][0].set_xdata(
                            [pos[j, 0], pos[j_parent, 0]]
                        )
                        lines_3d[n][j - 1][0].set_ydata(
                            [pos[j, 1], pos[j_parent, 1]]
                        )
                        lines_3d[n][j - 1][0].set_3d_properties(
                            [pos[j, 2], pos[j_parent, 2]], zdir="z"
                        )
                    else:
                        n_hyp = len(pos)
                        for hyp_idx, hyp in enumerate(pos):
                            multi_hyp_j_idx = (j - 1) * n_hyp + hyp_idx
                            lines_3d[n][multi_hyp_j_idx][0].set_xdata(
                                [hyp[j, 0], hyp[j_parent, 0]]
                            )
                            lines_3d[n][multi_hyp_j_idx][0].set_ydata(
                                [hyp[j, 1], hyp[j_parent, 1]]
                            )
                            lines_3d[n][multi_hyp_j_idx][0].set_3d_properties(
                                [hyp[j, 2], hyp[j_parent, 2]], zdir="z"
                            )
                            lines_3d[n][multi_hyp_j_idx][0].set_alpha(
                                hyp[j, 3] * 0.5 + 0.5 if hyp[j, 3] > 0.01 else 0.
                            )
                            # if j == 1:
                            #     lines_3d[n][multi_hyp_j_idx][0].set_label(
                            #         f"{hyp[j, 3]:.2f}"
                            #     )
                            # ax.legend(loc="lower center", ncol=3)

            render_animation.points.set_offsets(keypoints[i])

    fig.tight_layout()

    update_video(frame_index)

    fig.savefig(output, bbox_inches="tight")
    plt.close()


def render_rotated_frame_prediction(
    frame_index,
    keypoints,
    poses,
    skeleton,
    azim_list,
    fps,
    bitrate,
    output,
    viewport,
    size=6,
    input_video_path="",
    input_video_skip=0,
    elev=15,
):
    """Creates pdf or png of predicted skeleton in 2D and 3D (in global
    frame)
    """

    plt.ioff()
    setup_style(fontsize=28, lw=2.0)
    fig = plt.figure(figsize=(size * len(poses), size))
    # fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, len(poses), 1)
    # ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title("Input")

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    azim = azim_list[0]
    for index, (title, data) in enumerate(poses.items()):
        if title != "Ground truth":
            ax = fig.add_subplot(1, len(poses), index + 2, projection="3d")
            ax.set_title(title)  # , pad=35
        # ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        # ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        if len(data.shape) == 3:
            trajectories.append(data[:, 0, [0, 1]])
        else:
            trajectories.append(data[:, 0, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path == "":
        # Black background
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0]), dtype="uint8"
        )
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

    render_animation.initialized = False
    render_animation.image = None
    render_animation.lines = []
    render_animation.points = None

    parents = skeleton.parents

    def update_video(k):
        # CHANGE: Frame hard-coded
        i = frame_index
        for n, ax in enumerate(ax_3d):
            # CHANGE: Mofify azimuth
            ax.view_init(elev=elev, azim=azim_list[k])
        
            ax.set_xlim3d(
                [
                    -radius / 2 + trajectories[n][i, 0],
                    radius / 2 + trajectories[n][i, 0],
                ]
            )
            ax.set_ylim3d(
                [
                    -radius / 2 + trajectories[n][i, 1],
                    radius / 2 + trajectories[n][i, 1],
                ]
            )

        # Update 2D poses
        if not render_animation.initialized:
            render_animation.image = ax_in.imshow(
                all_frames[i], aspect="equal"
            )

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    # Draw skeleton only if keypoints match
                    # (otherwise we don't have the parents definition)
                    render_animation.lines.append(
                        ax_in.plot(
                            [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                            color="b",
                        )
                    )

                # col = "red" if j in skeleton.joints_right else "black"
                cpal = sns.color_palette("Dark2")
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    gt = poses[-1][i]
                    # Ground-truth
                    lines_3d[n].append(
                        ax.plot(
                            [gt[j, 0], gt[j_parent, 0]],
                            [gt[j, 1], gt[j_parent, 1]],
                            [gt[j, 2], gt[j_parent, 2]],
                            zdir="z",
                            c="black",
                        )
                    )
                    # Prediction
                    if len(pos.shape) == 2:
                        lines_3d[n].append(
                            ax.plot(
                                [pos[j, 0], pos[j_parent, 0]],
                                [pos[j, 1], pos[j_parent, 1]],
                                [pos[j, 2], pos[j_parent, 2]],
                                zdir="z",
                                c=cpal[0],
                            )
                        )
                    else:
                        for hyp, col in zip(pos, cpal):
                            lines_3d[n].append(
                                ax.plot(
                                    [hyp[j, 0], hyp[j_parent, 0]],
                                    [hyp[j, 1], hyp[j_parent, 1]],
                                    [hyp[j, 2], hyp[j_parent, 2]],
                                    zdir="z",
                                    c=col,
                                    alpha=hyp[j, 3] * 0.5 + 0.5 if hyp[j, 3] > 0.01 else 0.,  # <-- hyp score
                                    # label=f"{hyp[j, 3]:.2f}" if j == 1 else None
                                )
                            )
                        # ax.legend(loc="lower center", ncol=3)

            render_animation.points = ax_in.scatter(
                keypoints[i].T[0],
                keypoints[i].T[1],
                5,
                color="red",
                edgecolors="white",
                zorder=10,
            )

            render_animation.initialized = True
        else:
            render_animation.image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    render_animation.lines[j - 1][0].set_data(
                        [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                        [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                    )

                # for n, ax in enumerate(ax_3d):
                #     pos = poses[n][i]
                #     gt = poses[-1][i]

                #     # Ground-truth
                #     lines_3d[n][j - 1][0].set_xdata(
                #         [gt[j, 0], gt[j_parent, 0]]
                #     )
                #     lines_3d[n][j - 1][0].set_ydata(
                #         [gt[j, 1], gt[j_parent, 1]]
                #     )
                #     lines_3d[n][j - 1][0].set_3d_properties(
                #         [gt[j, 2], gt[j_parent, 2]], zdir="z"
                #     )

                #     # Prediction
                #     if len(pos.shape) == 2:
                #         lines_3d[n][j - 1][0].set_xdata(
                #             [pos[j, 0], pos[j_parent, 0]]
                #         )
                #         lines_3d[n][j - 1][0].set_ydata(
                #             [pos[j, 1], pos[j_parent, 1]]
                #         )
                #         lines_3d[n][j - 1][0].set_3d_properties(
                #             [pos[j, 2], pos[j_parent, 2]], zdir="z"
                #         )
                #     else:
                #         n_hyp = len(pos)
                #         for hyp_idx, hyp in enumerate(pos):
                #             multi_hyp_j_idx = (j - 1) * n_hyp + hyp_idx
                #             lines_3d[n][multi_hyp_j_idx][0].set_xdata(
                #                 [hyp[j, 0], hyp[j_parent, 0]]
                #             )
                #             lines_3d[n][multi_hyp_j_idx][0].set_ydata(
                #                 [hyp[j, 1], hyp[j_parent, 1]]
                #             )
                #             lines_3d[n][multi_hyp_j_idx][0].set_3d_properties(
                #                 [hyp[j, 2], hyp[j_parent, 2]], zdir="z"
                #             )
                #             lines_3d[n][multi_hyp_j_idx][0].set_alpha(
                #                 hyp[j, 3] * 0.5 + 0.5 if hyp[j, 3] > 0.01 else 0.
                #             )
                #             # if j == 1:
                #             #     lines_3d[n][multi_hyp_j_idx][0].set_label(
                #             #         f"{hyp[j, 3]:.2f}"
                #             #     )
                #             # ax.legend(loc="lower center", ncol=3)

            render_animation.points.set_offsets(keypoints[i])

    fig.tight_layout()

    anim = FuncAnimation(
        fig,
        update_video,
        frames=len(azim_list),
        interval=1000 / fps,
        repeat=False,
    )
    if output.endswith(".mp4"):
        Writer = writers["ffmpeg"]
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith(".gif"):
        anim.save(output, dpi=80, writer="imagemagick")
    else:
        raise ValueError(
            "Unsupported output format (only .mp4 and .gif are supported)"
        )
    plt.close()
