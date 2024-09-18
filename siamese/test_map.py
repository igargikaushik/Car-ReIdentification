import torch
import torch.nn.functional as F
import time

test_data = torch.load('./test_binary')
test_label = torch.load('./test_label')


def eucldist_generator(coords1, coords2):
    """ to calc euclidean dist b/w 2 lists of coordinates."""
    return sum((x - y)**2 for x, y in zip(coords1, coords2))**0.5

def cal_map_recall():
    str1 = []

    right_nums = 0
    retrieval_nums = 0
    for i in range(1000):
        start_time = time.time()
        for j in range(1000):
            if i == j:
                continue
            else:
                dis = eucldist_generator(test_data[i], test_data[j])
            if dis < 1 and i//10 == j//10:
                right_nums+=1
            if dis < 1:
                retrieval_nums+=1
        print("nums:",i,"time:",time.time()-start_time)
    print("accuracy",right_nums/retrieval_nums)
    print("recall",right_nums/9000)

    print(retrieval_nums)

if __name__== "__main__":
    print(test_data[0])
    dis = eucldist_generator(test_data[0],test_data[100])
    cal_map_recall()