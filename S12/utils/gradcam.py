import torch
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np


class GradCam(object):

    def __init__(self, model, target_layers, num_classes):
        super(GradCam, self).__init__()
        self.model = model
        self.target_layers = target_layers
        self.num_classes = num_classes
        self.device = next(model.parameters()).device

        self.activations_map = {}
        self.gradients_map = {}

        self.model.eval()
        self.register_hooks()

    def register_hooks(self):
        def _wrap_forward_hook(layer_name):
            def _forward_hook(module, input, output):
                self.activations_map[layer_name] = output.detach()

            return _forward_hook

        def _wrap_backward_hook(layer_name):
            def _backward_hook(module, grad_out, grad_in):
                self.gradients_map[layer_name] = grad_out[0].detach()

            return _backward_hook

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(_wrap_forward_hook(name))
                module.register_backward_hook(_wrap_backward_hook(name))

    def make_one_hots(self, target_class=None):
        one_hots = torch.zeros_like(self.output)
        if target_class:
            ids = torch.LongTensor([[target_class]] * self.batch_size).to(self.device)
            one_hots.scatter_(1, ids, 1.0)
        else:
            one_hots = torch.zeros((self.batch_size, self.num_classes)).to(self.device)
            for i in range(len(self.pred)):
                one_hots[i][self.pred[i][0]] = 1.0
        return one_hots

    def forward(self, data):
        self.batch_size, self.img_ch, self.img_h, self.img_w = data.shape
        data = data.to(self.device)
        self.output = self.model(data)
        self.pred = self.output.argmax(dim=1, keepdim=True)

    def backward(self, target_class=None):
        one_hots = self.make_one_hots(target_class)
        self.model.zero_grad()
        self.output.backward(gradient=one_hots, retain_graph=True)

    def __call__(self, data, target_layers, target_class=None):
        self.forward(data)
        self.backward(target_class)

        output = self.output
        saliency_maps = {}
        for target_layer in target_layers:
            activations = self.activations_map[target_layer]  # [64, 512, 4, 4]
            grads = self.gradients_map[target_layer]  # [64, 512, 4, 4]
            weights = F.adaptive_avg_pool2d(grads, 1)  # [64, 512, 1, 1]

            saliency_map = torch.mul(activations, weights).sum(dim=1, keepdim=True)
            saliency_map = F.relu(saliency_map)  # [64,1,4,4]
            saliency_map = F.interpolate(saliency_map, (self.img_h, self.img_w),
                                         mode="bilinear", align_corners=False)  # [64,1,32,32]

            saliency_map = saliency_map.view(self.batch_size, -1)
            saliency_map -= saliency_map.min(dim=1, keepdim=True)[0]
            saliency_map /= saliency_map.max(dim=1, keepdim=True)[0]
            saliency_map = saliency_map.view(self.batch_size, 1,
                                             self.img_h, self.img_w)
            saliency_maps[target_layer] = saliency_map

        return saliency_maps, self.pred


class VisualizeCam(object):

    def __init__(self, model, classes, target_layers):
        super(VisualizeCam, self).__init__()
        self.model = model
        self.classes = classes
        self.target_layers = target_layers
        self.device = next(model.parameters()).device

        self.gcam = GradCam(model, target_layers, len(classes))

    def visualize_cam(self, mask, img):
        heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b])

        result = heatmap + img.cpu()
        result = result.div(result.max()).squeeze()
        return heatmap, result

    def denormalize(self, tensor, mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]):
        single_img = False
        if tensor.ndimension() == 3:
            single_img = True
            tensor = tensor[None, :, :, :]

        if not tensor.ndimension() == 4:
            raise TypeError('tensor should be 4D')

        mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
        std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
        ret = tensor.mul(std).add(mean)
        return ret[0] if single_img else ret

    def plot_heatmaps(self, img_data, len_data, img_name,num_target=4):
        fig, axs = plt.subplots(nrows=len_data, ncols=num_target+1, figsize=(10, 2 * len_data),
                                subplot_kw={'xticks': [], 'yticks': []})
        #my_suptitle = fig.suptitle('GradCam at different conv layers', fontsize=13, weight='medium') #, y=1.05

        for ax, data in zip(axs.flat, img_data):
            img = data["img"]
            npimg = img.cpu().numpy()
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title("%s" % (data["label"]))

        fig.tight_layout()
        fig.subplots_adjust(top=0.97)
        fig.savefig(img_name)#,bbox_inches='tight',bbox_extra_artists=[my_suptitle]) #dpi=fig.dpi

    def __call__(self, images, images_gradcam, target_layers, target_inds=None, metric=""):
        masks_map, pred = self.gcam(images, target_layers, target_inds)

        heatmaps_data = []
        for i in range(len(images)):
            grad_img = images_gradcam[i]
            pred_class = self.classes[grad_img[1].data]
            act_class = self.classes[grad_img[2].data]
            img = images[i]

            heatmaps_data.append({
                "img": self.denormalize(img),
                "label": "Predicted: " + pred_class + "(Actual: " + act_class + ")"
            })
            for layer in target_layers:
                mask = masks_map[layer][i]
                heatmap = self.visualize_cam(mask, img)
                heatmaps_data.append({
                    "img": heatmap[0],
                    "label": layer
                })

        fname = "gradcam"
        self.plot_heatmaps(heatmaps_data, len(images), fname,len(target_layers))
