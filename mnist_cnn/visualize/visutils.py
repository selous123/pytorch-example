from visdom import Visdom
viz = Visdom()
import time
import torch
# win = viz.line(
#     X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
#     Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
# )

# time.sleep(20)

# viz.line(
#     X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
#     Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
#     win=win,
#     update='append'
# )
def visualize_loss(step,loss,env,win):
    """
    visualize loss
    Args:
        step :X value
        loss :loss value,tensor value
        env  :env of the loss in visdom
        win  :win of the loss in visdom
    """
    if win:
        viz.line(
            X = torch.Tensor([step]),
            Y = loss,
            env=env,
            win=win,
            update = 'append'
                )
    else:
        win=viz.line(
            X = torch.Tensor([step]),
            Y = loss,
            env=env
                )
    return win



if __name__=="__main__":
    t = torch.Tensor([1,2,3,4,5,6,7,8,9])
    loss = torch.Tensor([5,4,2,14,5,6,8,7,2])
    
    print loss[0:1]
    win=None
    for i in range(1,len(t)):
        win=visualize_loss(1,i,loss[i:i+1],env="malware",win=win)
        time.sleep(1)
# =============================================================================
#     for i in range(1,len(t)):
#         viz.line(
#             X = t[i:i+1],
#             Y = loss[i:i+1],
#             win = win,
#             update='append'
#         )
#         time.sleep(1)
# =============================================================================

