import multiprocessing
import logging as log
import threading
import time
import concurrent.futures


def calculate(x):
    log.info(f"In Calculate for X:{x}")
    time.sleep(2)
    return x ^ 2


if __name__ == "__main__":
    fmt = "%(asctime)s: %(message)s"
    log.basicConfig(format=fmt, level=log.INFO, datefmt="%H:%M:%S")
    # log.info("Before...")
    # x = threading.Thread(target=calculate, args=(2,))
    # log.info("Before Thread start")
    # x.start()
    # log.info("All Done")
    # # x.join()
    # log.info("Finally done after Join")
    #
    # log.info("Starting Y")
    # y = threading.Thread(target=calculate, args=(3,))
    # log.info("Started Y")
    # y.start()
    # log.info("Y started with start and join")
    # #y.join()
    # log.info("Y join done")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
        e.map(calculate, range(2))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        print(executor.submit(calculate, (2,)))
        executor.submit(calculate, (500,))
