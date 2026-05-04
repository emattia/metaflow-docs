
# Using GPUs and Other Accelerators

Metaflow enables access to hardware-accelerated computing, GPUs in particular, when
using [AWS Batch](aws-batch) or [Kubernetes](kubernetes).
You can leverage

- Single accelerators - e.g. `@resources(gpu=1)`
- Single instances with multiple accelerators - e.g. `@resources(gpu=4)`
- [Multiple instances with multiple accelerators](distributed-computing)

You can find many examples of how to use Metaflow to fine-tune LLMs and
other generative AI models, as well as how to train computer
vision and other deep learning models [in these articles](https://outerbounds.com/blog/?category=Foundation%20Models).

## Using accelerators

Before you can start taking advantage of hardware-accelerated steps, you need
to take care of two prerequisites:

1. Add hardware-accelerated instances in [your Metaflow stack](/getting-started/infrastructure).
Take a look specific tips for [AWS Batch](aws-batch) and [Kubernetes](kubernetes).

2. Configure your flow to [include necessary drivers and frameworks](installing-drivers-and-frameworks).

After this, using the accelerators in straightforward as explained below.

:::tip
Don't hesitate to reach out to [Metaflow Slack](http://chat.metaflow.org) if you need
help get started!
:::

### GPUs

To use GPUs in your compute environment, use [the
`@resources` decorator](requesting-resources) to get quick access to one or more GPUs
like in this example:

```python
from metaflow import FlowSpec, step, resources

class GPUFlow(FlowSpec):

    @resources(memory=32000, cpu=4, gpu=1)
    @step
    def start(self):
        from my_script import my_gpu_routine
        my_gpu_routine()
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    GPUFlow()
```

As usual with `@resources`, the decorator is ignored for local runs. This allows you to
develop the code locally, e.g. using GPU resources on your local workstation. To get access
to the requested resources in the cloud, run the flow `--with kubernetes` or `--with batch`.

If you need more fine-grained control over what GPUs get used, use the decorators
specific to compute environment: For instance, [`@kubernetes` allows you to
specify a `gpu_vendor`](/api/step-decorators/kubernetes) and [`@batch` allows you to
specify a `queue`](/api/step-decorators/batch) targeting a compute environment containing
specific GPUs. For more information, see guidance for [AWS Batch](aws-batch) and [Kubernetes](kubernetes).


### Using AWS Trainium and Inferentia

On AWS, you can use AWS-specific hardware accelerators, Trainium and Inferentia.
For more details, see [a blog post outlining them in the context of Metaflow](https://aws.amazon.com/blogs/machine-learning/develop-and-train-large-models-cost-efficiently-with-metaflow-and-aws-trainium/).

When using AWS Batch, you can request the accelerators by defining the number
of Trainium or Inferentia chips in `@batch`:

* `@batch(trainium=16)`
* `@batch(inferentia=16)`

When using `@kubernetes`, the same `trainium` parameter requests the corresponding
number of Neuron devices from the cluster. Both Trainium and Inferentia chips
register under a single Kubernetes resource — `aws.amazon.com/neuron` — managed
by the [AWS Neuron device plugin](https://github.com/aws-neuron/aws-neuron-k8s-device-plugin):

* `@kubernetes(trainium=1, image="public.ecr.aws/neuron/pytorch-training-neuronx:...")` — Trainium
* `@kubernetes(trainium=1, image="public.ecr.aws/neuron/pytorch-inference-neuronx:...")` — Inferentia

The cluster needs the Neuron device plugin DaemonSet running on Neuron-labeled nodes,
and the nodegroup's instance type must be a Neuron family member (`trn1`, `trn1n`,
`trn2.48xlarge`, `inf1`, `inf2`). Note that as of 2026, only `trn1.*`, `trn1n.32xlarge`,
and `inf1`/`inf2` are broadly available on-demand across multiple regions; `trn2.48xlarge`
is currently only available via [EC2 Capacity Blocks for ML](https://aws.amazon.com/ec2/capacityblocks/)
in the `us-east-2` region.

#### High-bandwidth networking with EFA

For multi-node training, Metaflow supports requesting [Elastic Fabric Adapter](https://aws.amazon.com/hpc/efa/)
network interfaces on Kubernetes via the `efa` parameter:

* `@kubernetes(gpu=8, efa=32)` — claim a full p5.48xlarge node with all 32 EFA NICs
* `@kubernetes(trainium=16, efa=8)` — full trn1.32xlarge node with 8 EFA NICs

EFA exposes `vpc.amazonaws.com/efa` as a schedulable resource managed by the
[AWS EFA k8s device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin).
Combined with NCCL via `aws-ofi-nccl`, this gives multi-node training the
high-bandwidth, low-latency RDMA path that AWS markets for these instance types.

Metaflow supports [distributed training](distributed-computing) over multiple
Trainium or GPU instances. For detailed instructions, visit
the [`metaflow-trainium` repository](https://github.com/outerbounds/metaflow-trainium/tree/main).


### Using Google's Tensor Processing Units (TPUs)

Contact [Metaflow Slack](http://chat.metaflow.org) if you are interested in using TPUs with
Metaflow in the Google cloud.

## Monitoring GPU utilization

To monitor GPU devices and their utilization, add [a custom card
`@gpu_profile`](https://github.com/outerbounds/metaflow-gpu-profile) in your GPU steps.

![](/assets/gpu_profile.png)

