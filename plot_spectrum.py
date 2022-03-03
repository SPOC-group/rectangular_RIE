import pandas as pd
import matplotlib.pyplot as plt

R1 = 2
R2 = .2
M = 5000

color = {"S" : (.27,.27,.86), "Noise" : (1,0,0), "Y"  : (1,.55,0), "Denoised" : (0,.39,0)}

plt.figure(figsize=(6.69,4))
plt.title(f"R2 = {R2}, R1 = {R1}")

Delta = 3e-1
plt.subplot(331)
df = pd.read_csv(f"data/SPECTRA/M{M}R1{R1}R2{R2}D{Delta}.csv")

plt.hist(df['Noise'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['Noise'])
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.xlim((-.1,4.5))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.ylabel(f'$\Delta = {Delta}$')

plt.subplot(332)
plt.hist(df['Y'], 160, density=True, histtype= "stepfilled", alpha=.7, color=color['Y'])
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.xlim((-.1,4.5))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.yticks([])

plt.subplot(333)
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.hist(df['Denoised'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['Denoised'])
plt.xlim((-.1,4.5))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.yticks([])


Delta = 3
plt.subplot(334)
df = pd.read_csv(f"data/SPECTRA/M{M}R1{R1}R2{R2}D{Delta}.csv")

plt.hist(df['Noise'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['Noise'])
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.xlim((-.1,5.2))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.ylabel(f'$\Delta = {Delta}$')

plt.subplot(335)
plt.hist(df['Y'], 160, density=True, histtype= "stepfilled", alpha=.7, color=color['Y'])
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.xlim((-.1,5.2))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.yticks([])

plt.subplot(336)
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.hist(df['Denoised'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['Denoised'])
plt.xlim((-.1,5.2))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.yticks([])


Delta = 30
plt.subplot(337)
df = pd.read_csv(f"data/SPECTRA/M{M}R1{R1}R2{R2}D{Delta}.csv")

plt.hist(df['Noise'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['Noise'])
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.xlim((-.1,15))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.xlabel('Noise')
plt.ylabel(f'$\Delta = {Delta}$')

plt.subplot(338)
plt.hist(df['Y'], 160, density=True, histtype= "stepfilled", alpha=.7, color=color['Y'])
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.xlim((-.1,15))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.xlabel('Observation')
plt.yticks([])


plt.subplot(339)
plt.hist(df['S'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['S'])
plt.hist(df['Denoised'], 160, density=True,  histtype= "stepfilled", alpha=.7, color=color['Denoised'])
plt.xlim((-.1,15))
plt.ylim((0.01,10.25))
plt.yscale('log')
plt.xlabel('Denoised')
plt.yticks([])


plt.subplots_adjust(left=.125, bottom=.11, right=.97, top=.95, wspace=.08, hspace=.274)
plt.savefig("spectrum_plot.pdf")
plt.show()
