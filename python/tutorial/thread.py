import threading
import time


class Worker(threading.Thread):

    def __init__(self, idx):
        super().__init__()
        self.worker_idx = idx

    def run(self):
        print('Worker {} Run'.format(self.worker_idx))
        time.sleep(0.1)
        print('Worker {} Finish'.format(self.worker_idx))


# Workers start at the same time, whoever finishes needs to wait for other workers to finish before printing out 'All done'

workers = []

for i in range(10):
    worker = Worker(i)
    worker.start()
    workers.append(worker)

[w.join() for w in workers]


print('All done')

# Workers start and finish one by one

for i in range(10):
    worker = Worker(i)
    worker.start()
    worker.join()

print('All done')
