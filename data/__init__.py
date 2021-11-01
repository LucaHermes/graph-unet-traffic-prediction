from data.dataset import T4CDataset

VOL_CHANNELS   = [0, 2, 4, 6]
SPEED_CHANNELS = [1, 3, 5, 7]

VOL_CHANNEL_SLICE = slice(0, 8, 2)
SPEED_CHANNEL_SLICE = slice(1, 8, 2)

CHANNEL_LABELS = {
	0 : 'vol. NE',
	2 : 'vol. SE',
	4 : 'vol. SW',
	6 : 'vol. NW',
	1 : 'speed NE',
	3 : 'speed SE',
	5 : 'speed SW',
	7 : 'speed NW'
}