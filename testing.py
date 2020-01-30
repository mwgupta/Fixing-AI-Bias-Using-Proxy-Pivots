# test function
from scipy.stats import entropy

plt_adv_loss = [1,2,3,4]
plt_clf_loss = [5,6,2,3]

print(min(plt_adv_loss))
print(min(min(plt_adv_loss),min(plt_clf_loss)))

