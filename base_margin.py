import torch
from torch import nn
from torchattacks.attack import Attack


def margin_metric(pred):
    topk_values, _ = torch.topk(pred, k=2)
    metric_value = topk_values[0][0] - topk_values[0][1]  # largest - second largest
    return metric_value


def mertric_sort(seed_num, plain_model, seed_loader, correct_tag=True, reverse=False):
    def metric_func(data_label_logits):
        data, label, logits = data_label_logits
        if logits is None:
            logits = plain_model(data)
        return margin_metric(logits).item()

    seedList = []
    if correct_tag:
        for data, label in seed_loader:
            outputs = plain_model(data)
            _, predicted = torch.max(outputs.data, 1)
            if predicted == label:
                seedList.append((data, label, outputs))
    else:
        seedList = [(data, label, None) for data, label in seed_loader]

    seedList.sort(key=metric_func)
    seedList = [(data, label) for data, label, logits in seedList]
    
    return seedList[:seed_num] if reverse == False else seedList[-seed_num:]


def uniform_noise(data, low=-0.03, high=0.03):
    return (high - low) * torch.rand(data.size()) + low


def normal_noise(data, mean=0, std=0.03):
    return torch.randn(data.size()) * std + mean


class MGPGD(Attack):
    def __init__(self, model, device=None, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__('MGPGD', model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels=None):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        alpha = self.alpha

        if self.random_start:
            adv_images = adv_images + \
                torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            cost =  -1 * margin_metric(outputs)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            alpha = alpha / 2
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        noise_images = adv_images - images

        return noise_images


class MGPGD_mu(Attack):
    def __init__(self, mutation_method, model, device=None, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__('MGPGD_mu', model)
        self.mutation_method = mutation_method
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels=None):
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        alpha = self.alpha

        if self.mutation_method == "margin":
            if self.random_start:
                adv_images = adv_images + \
                    torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.get_logits(adv_images)

                cost =  -1 * margin_metric(outputs)

                grad = torch.autograd.grad(cost, adv_images,
                                        retain_graph=False, create_graph=False)[0]

                adv_images = adv_images.detach() + alpha*grad.sign()
                alpha = alpha / 2
                delta = torch.clamp(adv_images - images,
                                    min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        elif self.mutation_method == "random":
            adv_images = adv_images + \
                         torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        elif self.mutation_method == "pgd":
            if self.targeted:
                target_labels = self.get_target_label(images, labels)

            loss = nn.CrossEntropyLoss()
            adv_images = images.clone().detach()

            if self.random_start:
                # Starting at a uniformly random point
                adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                    -self.eps, self.eps
                )
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()

            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.get_logits(adv_images)

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels.long())

                # Update adversarial images
                grad = torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]

                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        noise_images = adv_images - images

        return noise_images