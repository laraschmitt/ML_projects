import pandas as pd
import imageio
import matplotlib.pyplot as plt

# load dataframes
fert = pd.read_csv('data/gapminder_total_fertility.csv', index_col=0)
life = pd.read_excel('data/gapminder_lifeexpectancy.xlsx', index_col=0)
pop = pd.read_excel('data/gapminder_population.xlsx', index_col=0)


# convert string in the col names to int
ncol = [int(x) for x in fert.columns]
fert.set_axis(axis=1, labels=ncol, inplace=True)

# convert both tables to long format
sfert = fert.stack()
slife = life.stack
spop = pop.stack()

# convert series to dataframe
dict = {'fertility': sfert, 'lifeexp': slife, 'population': spop}
df_hierach_all = pd.DataFrame(data=dict)
df_hierach_all.iloc[95:220, :]

# stack and unstack
df3 = df_hierach_all.stack()

df6 = df3.unstack(1)
df6 = df6[2000]
df6 = df6.unstack(1)


for year in range(1920, 2016):

    df7 = df3.unstack(1)
    df_single_year_long_f = df7[year]  # all three characteristics for one year; long format
    df_single_year_wide_f = df_single_year_long_f.unstack(1)  # wide format
    df_single_year_wide_f.plot.scatter('fertility', 'lifeexp', s=df6['population']/10000000)

    # in one line
    #df3.unstack(1)[year].unstack(1).plot.scatter('fertility', 'lifeexp', s=df6['population']/10000000)

    plt.axis((0.5, 8, 9, 85))
    plt.title(str(year))

    plt.savefig(f'animated_plot/lifeexp_{year}.png')

images = []
for i in range(1960, 2015):
    filename = 'animated_plot/lifeexp_{}.png'.format(i)
    images.append(imageio.imread(filename))

imageio.mimsave('output.gif', images, fps=20)
