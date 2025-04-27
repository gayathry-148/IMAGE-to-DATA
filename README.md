# 'IMAGE-to-DATA'
# **Abstract** 
In the e-commerce domain, extracting meaningful and structured data from product 
images is crucial for enhancing digital storefronts. Many products lack detailed textual 
descriptions, making it necessary to rely on image-based information for key attributes such as 
weight, volume, and dimensions. This research addresses the problem by developing a robust 
machine learning pipeline that integrates Optical Character Recognition (OCR) with deep 
learning techniques. 
The proposed solution utilizes a hybrid model combining Convolutional Neural 
Networks (CNNs) for visual feature extraction and BERT-based embeddings for textual context 
derived from OCR. The model was trained on a custom dataset featuring various product 
categories and attributes, followed by preprocessing steps to ensure data consistency. A multi
task learning approach was adopted, predicting numerical values and their associated units 
simultaneously. 
Evaluation on test datasets yielded an F1 score of 0.73, demonstrating the model’s 
effectiveness in extracting complex attributes from diverse and noisy image data. Compared to 
conventional methods relying solely on OCR, the hybrid approach significantly improves 
precision and recall. This research highlights the potential of integrating multimodal data for 
image understanding tasks and provides a scalable framework for automating product 
information extraction, catering to the growing demands of digital marketplaces. 

#**1. Introduction **
As digital marketplaces expand, the demand for automated solutions to extract 
structured information from unstructured data grows significantly. Many e-commerce 
platforms rely heavily on product images to provide key details like weight, dimensions, 
wattage, and volume, especially for items lacking comprehensive textual descriptions. These 
details are essential for accurate product categorization, searchability, and improved user 
experience. However, manual data entry is not only time-consuming but also prone to errors, 
making automation a necessity. 
The complexity of extracting meaningful data from images stems from diverse 
challenges, including varying image quality, inconsistent formats, and text occlusion. 
Traditional Optical Character Recognition (OCR) systems are useful for extracting raw text but 
fail to provide the context or accurately predict structured values, especially when multiple 
attributes are present. Addressing this gap requires a solution that can handle multimodal 
inputs, combining both visual and textual information effectively. 
In this research, we propose a hybrid machine learning approach that integrates CNN
based image feature extraction with BERT-based text embeddings to accurately predict entity 
values and their corresponding units from product images. The model is designed to work 
across various product categories and handle the noisy data typical in real-world scenarios. By 
achieving an F1 score of 0.73, our solution demonstrates significant potential for transforming 
image-based data extraction processes in e-commerce.
