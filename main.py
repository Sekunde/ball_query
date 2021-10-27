import torch
import ball_query
pointsA = torch.FloatTensor([[[0,0,0]]]).cuda()
pointsB = torch.FloatTensor([[[10,10,10], [1,1,1], [2,2,2]]]).cuda()
return_idx = ball_query.ball_query(pointsA, pointsB, 5, 10)
print(return_idx)
