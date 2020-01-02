import flowiz as fz
import matplotlib.pyplot as plt

STEP_SIZE = 8
N_IMAGES = 2


def visualize(group: str):
    flow_path_template: str = 'dataset/flow/' + group + '-img{}.flo'
    for i in range(N_IMAGES):
        f = fz.read_flow(flow_path_template.format(i))

        u = f[::STEP_SIZE, ::STEP_SIZE, 0]
        v = f[::STEP_SIZE, ::STEP_SIZE, 1]
        plt.quiver(u, v, color='orange')
        plt.savefig(group + '-{}-vector-field.pdf'.format(i))
        plt.close()


if __name__ == '__main__':
    visualize('control')
    visualize('rotate')
