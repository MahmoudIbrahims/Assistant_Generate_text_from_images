from model import llava_model
from preprocessing import load_image,process_image
from preprocessing import create_prompt,ask_image,to_markdown
from translation import transArabic,transEnglish
from Text2Speech import text2speech,save_speech,play_audio


tokenizer, model, image_processor, context_len = llava_model()

image = load_image("bike-girl.jpeg")


processed_image = process_image(image,image_processor,model)
type(processed_image), processed_image.shape


prompt, _ = create_prompt("Describe the image")
print(prompt)


%%time
response = ask_image(image, "Describe the image",tokenizer, model)

reslut =transArabic(response)

to_markdown(reslut)

audio =text2speech(reslut)

save_speech(audio,"output.wav")

play_audio("output.wav")



