import numpy as np
import joblib
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

import soundfile as sf
from typing import List

import torchaudio
from torchaudio.functional import resample
import traceback
from peft import PeftModel
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

import fairseq
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder


NAME="SpeechGPT"
META_INSTRUCTION="""
    You are an AI assistant whose name is SpeechGPT.
    - SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University.
    SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.
    - It can perceive cross-modal inputs and generate cross-modal outputs.
"""
DEFAULT_GEN_PARAMS = {
    "max_new_tokens": 1024,
    "min_new_tokens": 10,
    "temperature": 0.8,
    "do_sample": True, 
    "top_k": 60,
    "top_p": 0.8,
}

def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response


class FeatureReader(object):
    def __init__(
        self,
        ckpt_path,
        layer,
        max_chunk=1600000,
        fp16=False,
        sampling_rate=16000,
        device="cpu",
    ):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()
        
        self.layer_shift = 0
        self.target_sample_hz = sampling_rate
        
        print(f"TASK CONFIG:\n{self.task.cfg}")

    def read_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        return wav

    @torch.no_grad()
    def get_feats(self, waveform):
        x = waveform
        with torch.no_grad():
            if self.fp16:
                x = x.half().cuda()
            else:
                x = x.float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer + self.layer_shift,
                )
        
                feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)


class ApplyKmeans:
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class Speech2Unit(torch.nn.Module):
    def __init__(
        self, 
        ckpt_dir,
        layer=11, 
        max_chunk=1600000, 
        fp16=False, 
        sampling_rate=16000,
        device="cpu",
        ):

        """
        Args:
            ckpt_dir(str): path to hubert model dir(e.g. hubert_base_ls960.pt)
            layer(int): feat from which layer of hubert models defauly by 9
            max_chunk(int): default by 1600000
            fp16(bool): default by False
            sampling_rate(int): sampling_rate default by 16000
        """
        super().__init__()
        self.device = torch.device(device)
        ckpt_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3.pt")
        km_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

        self.feature_reader = FeatureReader(ckpt_path, layer, max_chunk, fp16, sampling_rate)
        self.apply_kmeans = ApplyKmeans(km_path)
    
    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list
    

    def __call__(self, path, merged=True):
        waveform = self.feature_reader.read_audio(path).to(self.device)
        
        feat = self.feature_reader.get_feats(waveform)
        cluster_ids = self.apply_kmeans(feat).tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

        merged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        unmerged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"

        if merged:
            return merged_units
        else:
            return unmerged_units
        # return {"continuous":feat,
        # "units":dup_cluster_list,
        # "duration":duration_list,
        # "unmerged_units":cluster_ids}


class SpeechGPTInference:
    def __init__(
        self, 
        model_name_or_path: str,
        lora_weights: str=None,
        s2u_dir: str="speechgpt/utils/speech2unit/",
        vocoder_dir: str="speechgpt/utils/vocoder/", 
        output_dir="speechgpt/output/",
        device = "cpu",
        ):
        self.device = torch.device(device)
        self.meta_instruction = META_INSTRUCTION
        self.template= "[Human]: {question} <eoh>. [SpeechGPT]: "


        #speech2unit
        self.s2u = Speech2Unit(ckpt_dir=s2u_dir)
        
        #model
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )

        if lora_weights is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.model.half()  

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        #tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left" 


        #generation
        self.generate_kwargs = DEFAULT_GEN_PARAMS


        #vocoder
        vocoder = os.path.join(vocoder_dir, "vocoder.pt")
        vocoder_cfg = os.path.join(vocoder_dir, "config.json")
        with open(vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(vocoder, vocoder_cfg).to(device)

        self.output_dir = output_dir


    def preprocess(
        self,
        raw_text: str,
    ):
        processed_parts = []
        for part in raw_text.split("is input:"):
            if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [".wav", ".flac", ".mp4"]:
                processed_parts.append(self.s2u(part.strip(), merged=True))
            else:
                processed_parts.append(part)
        processed_text = "is input:".join(processed_parts)

        prompt_seq = self.meta_instruction + self.template.format(question=processed_text)
        return prompt_seq


    def postprocess(
        self,
        response: str,
    ):

        question = extract_text_between_tags(
            response, tag1="[Human]", tag2="<eoh>"
        )
        answer = extract_text_between_tags(
            response + '<eoa>', tag1=f"[SpeechGPT] :", tag2="<eoa>"
        )
        tq = extract_text_between_tags(
            response, tag1="[SpeechGPT] :", tag2="; [ta]"
        ) if "[ta]" in response else ''

        ta = extract_text_between_tags(
            response, tag1="[ta]", tag2="; [ua]"
        ) if "[ta]" in response else ''
        ua = extract_text_between_tags(response + '<eoa>', tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ''

        return {"question":question, "answer":answer, "textQuestion":tq, "textAnswer":ta, "unitAnswer":ua}


    def forward(
        self, 
        prompts: List[str]
    ):
        with torch.no_grad():
            #preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(self.preprocess(prompt))

            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
            for input_id in input_ids:
                if input_id[-1] == 2:
                    input_id = input_id[:, :-1]

            input_ids = input_ids.to(self.device)

            #generate
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                do_sample=True, 
                max_new_tokens=2048,
                min_new_tokens=10,
                )

            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                # max_new_tokens=1024,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)

            #postprocess
            responses = [self.postprocess(x) for x in responses]

            #save repsonses
            # init_num = sum(1 for line in open(f"{self.output_dir}/responses.json", 'r')) if os.path.exists(f"{self.output_dir}/responses.json") else 0             
            # with open(f"{self.output_dir}/responses.json", 'a') as f:
            for r in responses:
                if r["textAnswer"] != "":
                    print("Transcript:", r["textQuestion"])
                    print("Text response:", r["textAnswer"])
                else:
                    print("Response:\n", r["answer"])
                # json_line = json.dumps(r)
                # f.write(json_line+'\n')

            #dump wav
            wavs = list()
            wav = torch.tensor(0)
            # os.makedirs(f"{self.output_dir}/wav/", exist_ok=True)
            for i, response in enumerate(responses):
                if response["answer"] != '' and '<sosp>' in response["answer"]:
                    unit = [int(num) for num in re.findall(r'<(\d+)>', response["answer"])]
                    x = {
                            "code": torch.LongTensor(unit).view(1, -1).to(self.device),
                        }
                    wav = self.vocoder(x, True)
                    wavs.append(torch.clone(wav).cpu().numpy())
                    # self.dump_wav(init_num+i, wav, prefix="answer")
                    # print(f"Speech repsonse is saved in {self.output_dir}/wav/answer_{init_num+i}.wav")
            # print(f"Response json is saved in {self.output_dir}/responses.json")

        return 16000, np.hstack(wavs)

    def dump_wav(self, sample_id, pred_wav, prefix):
        sf.write(
            f"{self.output_dir}/wav/{prefix}_{sample_id}.wav",
            pred_wav.detach().cpu().numpy(),
            16000,
        )
        
    def __call__(self, input):
        return self.forward(input)

    
    def interact(self):
        prompt = str(input(f"Please talk with {NAME}:\n"))
        while prompt != "quit":
            try:
                self.forward([prompt])
            except Exception as e:
                traceback.print_exc()
                print(e)

            prompt = str(input(f"Please input prompts for {NAME}:\n"))
