"""
이미지 처리 유틸리티 모듈
프롬프트 정보 추출과 관련된 함수들을 포함 (버전 C - 통합 추출 로직)
"""
import gzip
import json
from typing import Optional, Tuple, Any, List
from PIL import Image, ExifTags
import piexif
import piexif.helper

TARGETKEY_NAIDICT_OPTION = ("steps", "height", "width",
                            "scale", "seed", "sampler", "n_samples", "sm", "sm_dyn",
                            "cfg scale", "cfg_scale", "clip skip", "clip_skip", "schedule type", "schedule_type",
                            "size", "model", "model hash", "model_hash", "denoising strength", "denoising_strength")

WEBUI_OPTION_MAPPING = {
    "cfg scale": "scale", "cfg_scale": "scale",
    "clip skip": "clip_skip", "clip_skip": "clip_skip",
    "schedule type": "schedule_type", "schedule_type": "schedule_type",
    "model hash": "model_hash", "model_hash": "model_hash",
    "denoising strength": "denoising_strength", "denoising_strength": "denoising_strength"
}

def is_nai_exif(info_str):
    if not info_str: return False
    try:
        data = json.loads(info_str)
        return 'Comment' in data and data['Comment'] is not None
    except Exception:
        return False

def parse_webui_exif(parameters_str):
    lines = parameters_str.splitlines()
    if not lines: return {}
    neg_prompt_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("Negative prompt:"):
            neg_prompt_index = i
            break

    if neg_prompt_index > 0:
        prompt = "\n".join(lines[:neg_prompt_index]).strip()
        negative_prompt = lines[neg_prompt_index][len("Negative prompt:"):].strip()
        option_lines = lines[neg_prompt_index+1:]
    else:
        prompt = "\n".join(lines).strip()
        negative_prompt = ""
        option_lines = []

    options, etc = {}, {}
    for line in option_lines:
        for part in line.split(','):
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                try:
                    value = float(value) if '.' in value else int(value)
                except: pass
                if key in WEBUI_OPTION_MAPPING: key = WEBUI_OPTION_MAPPING[key]
                if key.lower() in [k.lower() for k in TARGETKEY_NAIDICT_OPTION]: options[key] = value
                else: etc[key] = value
            elif part:
                etc[part] = ""

    return {"prompt": prompt, "uc": negative_prompt, "negative_prompt": negative_prompt, **options, **etc}

def _get_exifdict_from_infostr(info_str):
    if not info_str: return None
    try:
        data = json.loads(info_str)
        if 'parameters' in data: return parse_webui_exif(data['parameters'])
        elif 'Comment' in data: return None
        else: return data
    except json.JSONDecodeError:
        if "Prompt:" in info_str or "Negative prompt:" in info_str or "Steps:" in info_str:
            try: return parse_webui_exif(info_str)
            except: return None
        return None
    except Exception: return None

def _get_naidict_from_exifdict(exif_dict):
    try:
        nai_dict, all_prompts, all_neg_prompts = {}, [], []
        main_prompt = (exif_dict.get("prompt") or "").strip()
        if main_prompt: all_prompts.append(main_prompt)
        neg_prompt = (exif_dict.get("uc") or exif_dict.get("negative_prompt") or "").strip()
        if neg_prompt: all_neg_prompts.append(neg_prompt)

        char_captions_v1 = exif_dict.get("char_captions")
        if isinstance(char_captions_v1, list):
            for caption in char_captions_v1:
                if isinstance(caption, dict):
                    if caption.get("prompt"): all_prompts.append(caption["prompt"].strip())
                    if caption.get("neg_prompt"): all_neg_prompts.append(caption["neg_prompt"].strip())

        unique_prompts = list(dict.fromkeys(all_prompts))
        unique_neg_prompts = list(dict.fromkeys(all_neg_prompts))
        nai_dict["prompt"] = ", ".join(unique_prompts)
        nai_dict["negative_prompt"] = ", ".join(unique_neg_prompts)
        return nai_dict
    except Exception: return None

def _get_infostr_from_img(img):
    exif_str, pnginfo_str = None, None
    if img.format in ["WEBP", "JPEG"]:
        try:
            exif_bytes = img.info.get("exif")
            if exif_bytes:
                exif_dict = piexif.load(exif_bytes)
                user_comment_bytes = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                if user_comment_bytes: exif_str = piexif.helper.UserComment.load(user_comment_bytes)
        except: pass

    if exif_str is None and img.info:
        if 'Comment' in img.info and isinstance(img.info['Comment'], str):
            exif_str = img.info['Comment']
        elif 'parameters' in img.info and isinstance(img.info['parameters'], str):
            exif_str = img.info['parameters']

    try: pnginfo_str = read_info_from_image_stealth(img)
    except: pass
    return exif_str, pnginfo_str

def get_naidict_from_img(img):
    exif, pnginfo = _get_infostr_from_img(img)
    if not exif and not pnginfo: return None, 0
    for info_str in [exif, pnginfo]:
        if is_nai_exif(info_str):
            try:
                data = json.loads(info_str)
                return _get_naidict_from_exifdict(json.loads(data['Comment'])), 3
            except: pass
    ed1 = _get_exifdict_from_infostr(exif)
    ed2 = _get_exifdict_from_infostr(pnginfo)
    if not ed1 and not ed2: return exif or pnginfo, 1
    nd1 = _get_naidict_from_exifdict(ed1) if ed1 else None
    nd2 = _get_naidict_from_exifdict(ed2) if ed2 else None
    if not nd1 and not nd2: return exif or pnginfo, 2
    return nd1 or nd2, 3

def _recursive_extract_text(obj: Any) -> List[str]:
    texts = []
    if isinstance(obj, dict):
        if obj.get("class_type") == "CLIPTextEncode" and "inputs" in obj:
            val = obj["inputs"].get("text")
            if isinstance(val, str): texts.append(val)
        for value in obj.values():
            texts.extend(_recursive_extract_text(value))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(_recursive_extract_text(item))
    elif isinstance(obj, str):
        trimmed = obj.strip()
        if (trimmed.startswith('{') and trimmed.endswith('}')) or (trimmed.startswith('[') and trimmed.endswith(']')):
            try: texts.extend(_recursive_extract_text(json.loads(trimmed)))
            except: texts.append(trimmed)
        elif len(trimmed) > 0:
            texts.append(trimmed)
    return texts

def _extract_all_text_from_image_info(info_dict: dict) -> str:
    all_found_texts = []
    exclude_keys = {"exif", "icc_profile", "photoshop", "jpeg_rescale", "jpeg_restart_interval"}
    for key, value in info_dict.items():
        if key in exclude_keys: continue
        all_found_texts.extend(_recursive_extract_text(value))
    return "\n".join(list(dict.fromkeys(all_found_texts))).strip()

def read_info_from_image(image_path: str) -> str:
    """[버그 수정] 단일 반환 로직을 제거하고, 찾을 수 있는 모든 텍스트를 모아서 반환"""
    extracted_texts = []
    try:
        with Image.open(image_path) as img:
            png_info = img.info

            # 1. 기본 WebUI parameters 원본 보존 (부정 프롬프트, 모델명 포함)
            prompt_info = png_info.get("parameters", "")
            if prompt_info and isinstance(prompt_info, str):
                extracted_texts.append(prompt_info)

            # 2. NovelAI 및 WebUI 파싱 데이터 (부정 프롬프트 포함 추가)
            nai_dict, _ = get_naidict_from_img(img)
            if nai_dict:
                if isinstance(nai_dict, dict):
                    if "prompt" in nai_dict: extracted_texts.append(str(nai_dict["prompt"]))
                    if "negative_prompt" in nai_dict: extracted_texts.append(str(nai_dict["negative_prompt"]))
                elif isinstance(nai_dict, str):
                    extracted_texts.append(nai_dict)

            # 3. 범용 추출 (ComfyUI 포함 모든 문자열 탐색)
            universal_text = _extract_all_text_from_image_info(png_info)
            if universal_text:
                extracted_texts.append(universal_text)

            # 4. EXIF UserComment (Fooocus, JPEG WebUI)
            if img.format in ["JPEG", "WEBP"]:
                try:
                    exif_data = piexif.load(img.info.get("exif", b""))
                    if exif_data and "Exif" in exif_data:
                        user_comment = exif_data["Exif"].get(piexif.ExifIFD.UserComment)
                        if user_comment:
                            decoded = piexif.helper.UserComment.load(user_comment)
                            if decoded.startswith("{"):
                                try: extracted_texts.append(json.loads(decoded).get("comment", decoded))
                                except: extracted_texts.append(decoded)
                            else:
                                extracted_texts.append(decoded)
                except: pass

            # 5. Stealth PNGInfo
            stealth_info = read_info_from_image_stealth(img)
            if stealth_info: 
                extracted_texts.append(stealth_info)

        # 수집된 모든 텍스트를 하나의 문자열로 결합 (키워드 검색 효율 극대화)
        unique_texts = list(dict.fromkeys(extracted_texts))
        return "\n".join(unique_texts)
    except Exception as e:
        return ""

def read_info_from_image_stealth(image: Image.Image) -> Optional[str]:
    width, height = image.size
    pixels = image.load()
    has_alpha = image.mode == 'RGBA'
    mode, compressed = None, False
    binary_data, buffer_a, buffer_rgb = '', '', ''
    index_a, index_rgb = 0, 0
    sig_confirmed, confirming_signature = False, True
    reading_param_len, reading_param, read_end = False, False, False

    for x in range(width):
        for y in range(height):
            if has_alpha:
                r, g, b, a = pixels[x, y]
                buffer_a += str(a & 1)
                index_a += 1
            else: r, g, b = pixels[x, y]
            buffer_rgb += str(r & 1) + str(g & 1) + str(b & 1)
            index_rgb += 3
            
            if confirming_signature:
                if index_a == 120:  # len('stealth_pnginfo') * 8
                    decoded_sig = bytearray(int(buffer_a[i:i + 8], 2) for i in range(0, len(buffer_a), 8)).decode('utf-8', errors='ignore')
                    if decoded_sig in {'stealth_pnginfo', 'stealth_pngcomp'}:
                        confirming_signature, sig_confirmed, reading_param_len = False, True, True
                        mode = 'alpha'
                        compressed = (decoded_sig == 'stealth_pngcomp')
                        buffer_a, index_a = '', 0
                    else: read_end = True; break
                elif index_rgb == 120:
                    decoded_sig = bytearray(int(buffer_rgb[i:i + 8], 2) for i in range(0, len(buffer_rgb), 8)).decode('utf-8', errors='ignore')
                    if decoded_sig in {'stealth_rgbinfo', 'stealth_rgbcomp'}:
                        confirming_signature, sig_confirmed, reading_param_len = False, True, True
                        mode = 'rgb'
                        compressed = (decoded_sig == 'stealth_rgbcomp')
                        buffer_rgb, index_rgb = '', 0
                    else: read_end = True; break
            elif reading_param_len:
                if mode == 'alpha' and index_a == 32:
                    param_len, reading_param_len, reading_param = int(buffer_a, 2), False, True
                    buffer_a, index_a = '', 0
                elif mode == 'rgb' and index_rgb == 33:
                    pop = buffer_rgb[-1]
                    param_len, reading_param_len, reading_param = int(buffer_rgb[:-1], 2), False, True
                    buffer_rgb, index_rgb = pop, 1
            elif reading_param:
                if mode == 'alpha' and index_a == param_len:
                    binary_data, read_end = buffer_a, True; break
                elif mode == 'rgb' and index_rgb >= param_len:
                    diff = param_len - index_rgb
                    if diff < 0: buffer_rgb = buffer_rgb[:diff]
                    binary_data, read_end = buffer_rgb, True; break
        if read_end: break

    if sig_confirmed and binary_data != '':
        byte_data = bytearray(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8))
        try:
            return gzip.decompress(bytes(byte_data)).decode('utf-8') if compressed else byte_data.decode('utf-8', errors='ignore')
        except: pass
    return None