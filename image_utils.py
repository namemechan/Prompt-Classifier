"""
이미지 처리 유틸리티 모듈
프롬프트 정보 추출과 관련된 함수들을 포함
"""
import gzip
import json
from typing import Optional, Tuple
from PIL import Image, ExifTags
import piexif
import piexif.helper

TARGETKEY_NAIDICT_OPTION = ("steps", "height", "width",
                            "scale", "seed", "sampler", "n_samples", "sm", "sm_dyn",
                            # WebUI options
                            "cfg scale", "cfg_scale", "clip skip", "clip_skip", "schedule type", "schedule_type",
                            "size", "model", "model hash", "model_hash", "denoising strength", "denoising_strength")

WEBUI_OPTION_MAPPING = {
    "cfg scale": "scale",
    "cfg_scale": "scale",
    "clip skip": "clip_skip",
    "clip_skip": "clip_skip",
    "schedule type": "schedule_type",
    "schedule_type": "schedule_type",
    "model hash": "model_hash",
    "model_hash": "model_hash",
    "denoising strength": "denoising_strength",
    "denoising_strength": "denoising_strength"
}

def is_nai_exif(info_str):
    """nai 이미지면 exif의 원본 JSON에 'Comment' 키가 존재하고 None이 아닌 경우 True를 반환"""
    if not info_str:
        return False
    try:
        data = json.loads(info_str)
        return 'Comment' in data and data['Comment'] is not None
    except Exception as e:
        return False

def parse_webui_exif(parameters_str):
    """
    WebUI EXIF의 'parameters' 문자열을 파싱합니다.
    """
    lines = parameters_str.splitlines()
    if not lines:
        return {}

    # Negative prompt 라인을 찾음
    neg_prompt_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("Negative prompt:"):
            neg_prompt_index = i
            break

    # 프롬프트 추출 (Negative prompt 전까지의 모든 줄)
    if neg_prompt_index > 0:
        prompt = "\n".join(lines[:neg_prompt_index]).strip()
        negative_prompt = lines[neg_prompt_index][len("Negative prompt:"):].strip()
        option_lines = lines[neg_prompt_index+1:]
    else:
        # Negative prompt가 없는 경우
        prompt = "\n".join(lines).strip()
        negative_prompt = ""
        option_lines = []

    options = {}
    etc = {}

    # 옵션 파싱
    for line in option_lines:
        line = line.strip()
        parts = line.split(',')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip().lower()
                value = value.strip()

                # 숫자 변환 시도
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except:
                    pass

                if key in WEBUI_OPTION_MAPPING:
                    key = WEBUI_OPTION_MAPPING[key]

                if key.lower() in [k.lower() for k in TARGETKEY_NAIDICT_OPTION]:
                    options[key] = value
                else:
                    etc[key] = value
            elif part:
                etc[part] = ""

    return {
        "prompt": prompt,
        "uc": negative_prompt,  # NAI 호환을 위해 "uc" 사용
        "negative_prompt": negative_prompt,  # WebUI 표준 키도 유지
        **options,  # 옵션 평탄화
        **etc  # 기타 필드 평탄화
    }

def _get_exifdict_from_infostr(info_str):
    if not info_str:
        return None
    try:
        data = json.loads(info_str)
        # WebUI 형식의 경우 'parameters' 키가 존재함
        if 'parameters' in data:
            return parse_webui_exif(data['parameters'])
        # nai 이미지라면 여기서 처리하지 않고 get_naidict_from_img에서 old 방식으로 처리함
        elif 'Comment' in data:
            return None
        else:
            return data
    except json.JSONDecodeError:
        # If it's not a valid JSON, it might be a raw WebUI parameter string
        # Check if it contains common WebUI parameter indicators
        if "Prompt:" in info_str or "Negative prompt:" in info_str or "Steps:" in info_str:
            try:
                return parse_webui_exif(info_str)
            except Exception as e:
                print(f"Error parsing raw WebUI string: {e}")
                return None
        else:
            # It's neither JSON nor a recognizable WebUI raw string
            print(f"EXIF dictionary conversion error: Not a valid JSON or WebUI string. Info: {info_str[:100]}...")
            return None
    except Exception as e:
        print("EXIF dictionary conversion error (general):", e)
        return None

def _get_naidict_from_exifdict(exif_dict):
    try:
        nai_dict = {}

        # 프롬프트 처리 (None인 경우 빈 문자열로)
        nai_dict["prompt"] = (exif_dict.get("prompt") or "").strip()

        # 네거티브 프롬프트 처리
        if "uc" in exif_dict and exif_dict.get("uc") is not None:
            nai_dict["negative_prompt"] = (exif_dict.get("uc") or "").strip()
        elif "negative_prompt" in exif_dict and exif_dict.get("negative_prompt") is not None:
            nai_dict["negative_prompt"] = (exif_dict.get("negative_prompt") or "").strip()
        else:
            nai_dict["negative_prompt"] = ""

        # 옵션 추출
        option_dict = {}
        for key in TARGETKEY_NAIDICT_OPTION:
            if key in exif_dict and exif_dict[key] is not None:
                option_dict[key] = exif_dict[key]

        # WebUI 옵션 매핑
        for webui_key, nai_key in WEBUI_OPTION_MAPPING.items():
            if webui_key in exif_dict and exif_dict[webui_key] is not None:
                option_dict[nai_key] = exif_dict[webui_key]

        nai_dict["option"] = option_dict

        # 기타 정보 처리
        etc_dict = {}
        excluded_keys = list(TARGETKEY_NAIDICT_OPTION) + ["prompt", "uc", "negative_prompt"]
        excluded_keys.extend(WEBUI_OPTION_MAPPING.keys())

        for key in exif_dict.keys():
            if key not in excluded_keys:
                etc_dict[key] = exif_dict[key]

        nai_dict["etc"] = etc_dict

        return nai_dict
    except Exception as e:
        print("Error in _get_naidict_from_exifdict:", e)
    return None

def _get_infostr_from_img(img):
    exif_str = None
    pnginfo_str = None

    # Try to get EXIF UserComment for WebP/JPEG
    if img.format in ["WEBP", "JPEG"]:
        try:
            exif_bytes = img.info.get("exif")
            if exif_bytes:
                exif_dict = piexif.load(exif_bytes)
                user_comment_bytes = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                if user_comment_bytes:
                    exif_str = piexif.helper.UserComment.load(user_comment_bytes)
        except Exception as e:
            print(f"Error reading WebP/JPEG EXIF UserComment: {e}")

    # Fallback to general img.info if no UserComment or for other formats
    if exif_str is None and img.info:
        try:
            # Check if img.info itself contains a 'Comment' or 'parameters' key directly
            # This handles cases where info is already a dict with relevant data
            if 'Comment' in img.info and img.info['Comment'] is not None:
                exif_str = img.info['Comment']
            elif 'parameters' in img.info and img.info['parameters'] is not None:
                exif_str = img.info['parameters']
            else:
                # If not directly a comment/parameters, try to dump the whole info dict
                exif_str = json.dumps(img.info)
        except Exception as e:
            print(f"Error dumping img.info to JSON: {e}")

    # stealth pnginfo (should work for RGBA WebP too)
    try:
        # This function is already defined in image_utils.py, so we will call the existing one.
        pnginfo_str = read_info_from_image_stealth(img)
    except Exception as e:
        print(f"Error reading stealth info: {e}")

    return exif_str, pnginfo_str

def get_naidict_from_img(img):
    # ComfyUI data parsing (commented out for now as comfyui_parser is not available)
    # if img.info and 'prompt' in img.info:
    #     try:
    #         comfyui_data = get_comfyui_data(img.info, img.width, img.height)
    #         if comfyui_data:
    #             return comfyui_data, 3
    #     except Exception as e:
    #         print(f"Error parsing ComfyUI data: {e}")

    exif, pnginfo = _get_infostr_from_img(img)
    if not exif and not pnginfo:
        return None, 0

    # 먼저 nai 이미지 여부를 검사하여, nai 이미지면 old 방식으로 처리
    for info_str in [exif, pnginfo]:
        if is_nai_exif(info_str):
            try:
                data = json.loads(info_str)
                nai_exif = json.loads(data['Comment'])
                nd = _get_naidict_from_exifdict(nai_exif)
                if nd:
                    return nd, 3
            except Exception as e:
                print(f"Error in nai old method extraction: {e}")

    # nai 이미지가 아니라면 WebUI 방식(new)으로 처리
    ed1 = _get_exifdict_from_infostr(exif)
    ed2 = _get_exifdict_from_infostr(pnginfo)
    if not ed1 and not ed2:
        return exif or pnginfo, 1

    nd1 = _get_naidict_from_exifdict(ed1) if ed1 else None
    nd2 = _get_naidict_from_exifdict(ed2) if ed2 else None
    if not nd1 and not nd2:
        return exif or pnginfo, 2

    if nd1:
        return nd1, 3
    else:
        return nd2, 3

def read_info_from_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            # NovelAI 이미지 정보 추출 시도
            nai_dict, _ = get_naidict_from_img(img)
            if nai_dict and "prompt" in nai_dict:
                return nai_dict["prompt"]

            # 방법 1: PNG 메타데이터 읽기 (parameters 항목)
            png_info = img.info
            prompt_info = png_info.get("parameters", "")
            if prompt_info:
                return prompt_info

            # 방법 2: ComfyUI PNG 메타데이터 읽기 (prompt 항목)
            comfyui_prompt = _extract_comfyui_prompt(png_info)
            if comfyui_prompt:
                return comfyui_prompt

            # 방법 3: EXIF 데이터에서 프롬프트 정보 추출 (JPEG, WEBP 등)
            if img.format in ["JPEG", "WEBP"]:
                try:
                    # piexif를 사용하여 EXIF 데이터 로드
                    exif_data = piexif.load(img.info.get("exif"))
                    if exif_data and "Exif" in exif_data:
                        user_comment = exif_data["Exif"].get(piexif.ExifIFD.UserComment)
                        if user_comment:
                            # piexif.helper.UserComment.load를 사용하여 디코딩
                            decoded_comment = piexif.helper.UserComment.load(user_comment)
                            # image_data_reader.py에서처럼 JSON 파싱 시도 (Fooocus, Easy Diffusion 등)
                            if decoded_comment.startswith("{") and decoded_comment.endswith("}"):
                                try:
                                    comment_json = json.loads(decoded_comment)
                                    # Fooocus의 경우 "comment" 키에 프롬프트가 있을 수 있음
                                    if "comment" in comment_json:
                                        return comment_json["comment"]
                                    # Easy Diffusion 등 다른 JSON 기반 프롬프트도 처리 가능
                                    return decoded_comment
                                except json.JSONDecodeError:
                                    pass # JSON이 아니면 일반 텍스트로 처리
                            return decoded_comment
                except Exception as e:
                    print(f"EXIF 데이터 읽기 오류: {str(e)}")
                    pass # EXIF가 없거나 오류 발생 시 다음 방법 시도

            # 방법 3: 기존 stealth pnginfo 읽기 (LSB 스테가노그래피)
            stealth_info = read_info_from_image_stealth(img)
            if stealth_info:
                return stealth_info

        return ""
    except Exception as e:
        print(f"이미지 읽기 오류: {str(e)}")
        return ""


def _extract_comfyui_prompt(image_info: dict) -> Optional[str]:
    try:
        prompt_json_str = image_info.get("prompt")
        if not prompt_json_str:
            return None

        prompt_data = None
        if isinstance(prompt_json_str, str):
            try:
                prompt_data = json.loads(prompt_json_str)
            except json.JSONDecodeError as e:
                print(f"ComfyUI JSON 디코딩 오류: {e}") # 디버깅용
                return None
        elif isinstance(prompt_json_str, dict):
            prompt_data = prompt_json_str
        else:
            print(f"ComfyUI 프롬프트 데이터 타입 오류: {type(prompt_json_str)}") # 디버깅용
            return None

        if not isinstance(prompt_data, dict):
            print(f"ComfyUI 파싱 후 프롬프트 데이터가 딕셔너리가 아님: {type(prompt_data)}") # 디버깅용
            return None

        found_prompts = []

        # 워크플로우의 노드들을 순회
        for node_id, node_data in prompt_data.items():
            class_type = node_data.get("class_type")
            inputs = node_data.get("inputs")

            # CLIPTextEncode 노드를 찾아 'text' 입력 추출
            if class_type == "CLIPTextEncode" and inputs and "text" in inputs:
                text = inputs["text"]
                if isinstance(text, str):
                    found_prompts.append(text)

        if found_prompts:
            return "\n".join(found_prompts)

    except Exception as e: # 예상치 못한 다른 오류를 잡기 위함
        print(f"ComfyUI 프롬프트 추출 중 예상치 못한 오류: {e}") # 디버깅용
        return None
    return None

def read_info_from_image_stealth(image: Image.Image) -> Optional[str]:
    """
    이미지에서 stealth pnginfo를 추출
    """
    width, height = image.size
    pixels = image.load()

    has_alpha = image.mode == 'RGBA'
    
    # 초기화
    mode = None
    compressed = False
    binary_data = ''
    buffer_a = ''
    buffer_rgb = ''
    index_a = 0
    index_rgb = 0
    
    # 상태 플래그
    sig_confirmed = False
    confirming_signature = True
    reading_param_len = False
    reading_param = False
    read_end = False
    param_len = 0
    
    # 픽셀별로 처리
    for x in range(width):
        for y in range(height):
            # 픽셀 값 가져오기
            if has_alpha:
                r, g, b, a = pixels[x, y]
                buffer_a += str(a & 1)
                index_a += 1
            else:
                r, g, b = pixels[x, y]
            
            buffer_rgb += str(r & 1)
            buffer_rgb += str(g & 1)
            buffer_rgb += str(b & 1)
            index_rgb += 3
            
            # 상태에 따른 처리
            if confirming_signature:
                if _check_signature(buffer_a, buffer_rgb, index_a, index_rgb, has_alpha):
                    sig_confirmed, confirming_signature, reading_param_len, mode, compressed, buffer_a, buffer_rgb, index_a, index_rgb = _process_signature(buffer_a, buffer_rgb, index_a, index_rgb, has_alpha)
            elif reading_param_len:
                if _is_param_len_ready(mode, index_a, index_rgb):
                    reading_param_len, reading_param, param_len, buffer_a, buffer_rgb, index_a, index_rgb = _process_param_len(mode, buffer_a, buffer_rgb, index_a, index_rgb)
            elif reading_param:
                if _is_param_ready(mode, index_a, index_rgb, param_len):
                    binary_data = buffer_a if mode == 'alpha' else buffer_rgb
                    read_end = True
                    break
            else:
                # 불가능한 상태
                read_end = True
                break
                
        if read_end:
            break
    
    # 데이터 디코딩
    if sig_confirmed and binary_data:
        return _decode_binary_data(binary_data, compressed)
    
    return None


def _check_signature(buffer_a: str, buffer_rgb: str, index_a: int, index_rgb: int, has_alpha: bool) -> bool:
    """시그니처 확인이 가능한지 체크"""
    if has_alpha and index_a == len('stealth_pnginfo') * 8:
        return True
    elif index_rgb == len('stealth_pnginfo') * 8:
        return True
    return False


def _process_signature(buffer_a: str, buffer_rgb: str, index_a: int, index_rgb: int, has_alpha: bool) -> Tuple:
    """시그니처 처리"""
    sig_confirmed = False
    confirming_signature = False
    reading_param_len = False
    mode = None
    compressed = False
    
    if has_alpha:
        decoded_sig = _decode_bytes(buffer_a)
        if decoded_sig in {'stealth_pnginfo', 'stealth_pngcomp'}:
            sig_confirmed = True
            reading_param_len = True
            mode = 'alpha'
            compressed = decoded_sig == 'stealth_pngcomp'
            buffer_a = ''
            index_a = 0
    else:
        decoded_sig = _decode_bytes(buffer_rgb)
        if decoded_sig in {'stealth_rgbinfo', 'stealth_rgbcomp'}:
            sig_confirmed = True
            reading_param_len = True
            mode = 'rgb'
            compressed = decoded_sig == 'stealth_rgbcomp'
            buffer_rgb = ''
            index_rgb = 0
    
    return sig_confirmed, confirming_signature, reading_param_len, mode, compressed, buffer_a, buffer_rgb, index_a, index_rgb


def _is_param_len_ready(mode: str, index_a: int, index_rgb: int) -> bool:
    """파라미터 길이를 읽을 준비가 되었는지 확인"""
    return (mode == 'alpha' and index_a == 32) or (mode == 'rgb' and index_rgb == 33)


def _process_param_len(mode: str, buffer_a: str, buffer_rgb: str, index_a: int, index_rgb: int) -> Tuple:
    """파라미터 길이 처리"""
    reading_param_len = False
    reading_param = True
    param_len = 0
    
    if mode == 'alpha':
        param_len = int(buffer_a, 2)
        buffer_a = ''
        index_a = 0
    else:
        pop = buffer_rgb[-1]
        buffer_rgb = buffer_rgb[:-1]
        param_len = int(buffer_rgb, 2)
        buffer_rgb = pop
        index_rgb = 1
    
    return reading_param_len, reading_param, param_len, buffer_a, buffer_rgb, index_a, index_rgb


def _is_param_ready(mode: str, index_a: int, index_rgb: int, param_len: int) -> bool:
    """파라미터를 읽을 준비가 되었는지 확인"""
    return (mode == 'alpha' and index_a == param_len) or (mode == 'rgb' and index_rgb >= param_len)


def _decode_bytes(binary_str: str) -> str:
    """바이너리 문자열을 문자열로 디코딩"""
    try:
        byte_data = bytearray(int(binary_str[i:i + 8], 2) for i in range(0, len(binary_str), 8))
        return byte_data.decode('utf-8', errors='ignore')
    except Exception:
        return ""


def _decode_binary_data(binary_data: str, compressed: bool) -> Optional[str]:
    """바이너리 데이터를 디코딩"""
    try:
        byte_data = bytearray(int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8))
        if compressed:
            return gzip.decompress(bytes(byte_data)).decode('utf-8')
        else:
            return byte_data.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"데이터 디코딩 오류: {str(e)}")
        return None