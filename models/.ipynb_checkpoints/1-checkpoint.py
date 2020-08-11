    def sailency_result(self, _input, _prob):

        batchsizenum, channel, row, col = _input.shape
        h = torch.zeros([batchsizenum, 1, row, col])
        for i in range(batchsizenum):
            prob = _prob[i, 1].reshape([row, col]).cpu()
            img = _input[i, 1].reshape([row, col]).cpu().numpy()
            S = np.copy(prob)
            S[prob > 0.5] = 1
            S = prob.numpy().astype(np.uint8)

            D1 = geodesic_distance.geodesic2d_fast_marching(img, S)
            D1 = D1/D1.max()
            D1 = 1-D1
            plt.figure(figsize=(15, 5))
            plt.subplot(2, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.subplot(2, 3, 2)
            plt.imshow(D1)
            #plt.subplot(2, 3, 3)
            #plt.imshow(img*D1, cmap='gray')

            D3 = geodesic_distance.geodesic2d_raster_scan(img, S, 0.0, 2)
            D3 = D3/D3.max()
            D3 = 1-D3
            #plt.figure(figsize=(15, 5))
            plt.subplot(2, 3, 3)
            plt.imshow(img, cmap='gray')
            plt.subplot(2, 3, 4)
            plt.imshow(D3)
            plt.subplot(2, 3, 5)
            plt.imshow(img*D3, cmap='gray')

            D1_tensor = torch.from_numpy(D1)
            h[i, 0] = D1_tensor.cuda()


        saliency = self.relu_saliency1(self.saliency1(h.cuda()))
        saliency = self.saliency3(saliency)


        #_input = _input*saliency

        D1 = _input[0, 0].data.cpu().numpy()
        D2 = _input[0, 1].data.cpu().numpy()
        D3 = _input[0, 2].data.cpu().numpy()
        D0 = _input[0].data.cpu().numpy()
        D0 = D0.transpose((1, 2, 0))
        plt.figure()
        plt.subplot(3,4,1)
        plt.imshow(D1, cmap='gray')
        plt.subplot(3,4,2)
        plt.imshow(D2, cmap='gray')
        plt.subplot(3,4,3)
        plt.imshow(D3, cmap='gray')
        plt.subplot(3,4,4)
        plt.imshow(D0)


        D11 = saliency[0, 0].data.cpu().numpy()
        D22 = saliency[0, 1].data.cpu().numpy()
        D33 = saliency[0, 2].data.cpu().numpy()
        D00 = saliency[0].data.cpu().numpy()
        D00 = D00.transpose((1, 2, 0))
        #plt.figure()
        plt.subplot(3,4,5)
        plt.imshow(D11, cmap='gray')
        plt.axis('off');
        plt.title('(a) channel 1 of saliency map')
        plt.subplot(3,4,6)
        plt.imshow(D22, cmap='gray')
        plt.axis('off');
        plt.title('(b) channel 2 of saliency map')
        plt.subplot(3,4,7)
        plt.imshow(D33, cmap='gray')
        plt.axis('off');
        plt.title('(c) channel 3 of saliency map')
        plt.subplot(3,4,8)
        plt.imshow(D00)
        plt.title('(c) saliency map')


       # plt.figure()
        plt.subplot(3,4,9)
        plt.imshow(D11*D1, cmap='gray')
        plt.axis('off');
        plt.title('(a) channel 1 of updated image')
        plt.subplot(3,4,10)
        plt.imshow(D22*D2, cmap='gray')
        plt.axis('off');
        plt.title('(a) channel 2 of updated image')
        plt.subplot(3,4,11)
        plt.imshow(D33*D3, cmap='gray')
        plt.axis('off');
        plt.title('(a) channel 3 of updated image')
        plt.subplot(3,4,12)
        plt.imshow(D0*D00)
        plt.axis('off');
        plt.title('(a) updated image')





        _input_pil = transforms.ToPILImage()(_input[0].cpu())
        _input_pil = transforms.ToPILImage()(saliency[0].cpu())
        _input_pil.show()
#

        return saliency, _input
        # print(_input.size())