import random



def freezingModifications(args, model, output_dir):
	if args.portion:
		PORTION = (args.portion / 100)
	else:
		PORTION = 1
	if args.plF:
		output_dir = output_dir + "plF/" + str(args.portion) + "/"
		layers = []

		for layer in model.bert.encoder.layer:
			layers.append(layer)

		count = int(len(layers) * PORTION)
		layerSubset = random.sample(layers, count)

		for layer in layerSubset:
			params = []
			for param in layer.parameters():
				params.append(param)
			count = int(len(params))
			paramSubset = random.sample(params, count)
			for param in paramSubset:
				param.requires_grad = False

	if args.freeze:
		output_dir = output_dir + "freeze/"
		for layer in model.bert.encoder.layer:
			for param in layer.parameters():
				param.requires_grad = False

	if args.layerF:
		output_dir = output_dir + "layerF/" + str(args.portion) + "/"
		layers = []
		for layer in model.bert.encoder.layer:
			layers.append(layer)

		count = int(len(layers) * PORTION)
		layers = random.sample(layers, count)

		for layer in layers:
			for param in layer.parameters():
				param.requires_grad = False

	if args.paramF:
		output_dir = output_dir + "paramF/" + str(args.portion) + "/"
		parameters = []
		for layer in model.bert.encoder.layer:
			for param in layer.parameters():
				parameters.append(param)

		for param in model.parameters():
			parameters.append(param)

		count = int(len(parameters) * PORTION)
		subset = random.sample(parameters, count)
		for param in subset:
			param.requires_grad = False

	return model, output_dir
