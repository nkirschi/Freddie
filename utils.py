def normalize_decoupled(data, cols):
    data[cols] = (data[cols] - data[cols].mean()) / data[cols].std()


def normalize_coupled(data, cols):
    centered = data[cols] - data[cols].mean()
    distdev = (centered ** 2).sum(axis=1).mean() ** 0.5
    data[cols] = centered / distdev