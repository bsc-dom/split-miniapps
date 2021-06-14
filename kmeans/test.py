
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.reduction import reduction
from pycompss.api.parameter import COLLECTION_IN


@reduction(chunk_size=48)
@task(returns=1, input=COLLECTION_IN)
def my_sum(input):
    return sum(input)


@task(returns=1)
def one():
    return 1

if __name__ == "__main__":
    partials = list()
    for i in range(96):
        partials.append(one())
    
    result = my_sum(partials)

    print("First result (96 tasks): %d" % compss_wait_on(result))

    partials = list()
    for i in range(192):
        partials.append(one())
    
    result = my_sum(partials)

    print("First result (192 tasks): %d" % compss_wait_on(result))
