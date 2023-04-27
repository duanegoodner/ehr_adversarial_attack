import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("common X")
plt.ylabel("common Y")

plt.show()