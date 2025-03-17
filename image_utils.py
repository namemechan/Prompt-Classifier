"""
이미지 처리 유틸리티 모듈
프롬프트 정보 추출과 관련된 함수들을 포함
"""
import gzip
from typing import Optional, Tuple
from PIL import Image, ExifTags

def read_info_from_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            # 방법 1: PNG 메타데이터 읽기
            png_info = img.info
            prompt_info = png_info.get("parameters", "")
            if prompt_info:
                return prompt_info

            # **추가**: EXIF 데이터에서 프롬프트 정보 추출
            exif_data = img.getexif()
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    # 일반적으로 프롬프트 정보가 담길 수 있는 태그 (필요에 따라 조정)
                    if tag_name in ["UserComment", "ImageDescription", "XPComment"]:
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except Exception:
                                continue
                        if value:
                            return value

            # 방법 2: stealth pnginfo 읽기 (기존 방식)
            stealth_info = read_info_from_image_stealth(img)
            if stealth_info:
                return stealth_info

        return ""
    except Exception as e:
        print(f"이미지 읽기 오류: {str(e)}")
        return ""


def read_info_from_image_stealth(image: Image.Image) -> Optional[str]:
    """
    이미지에서 stealth pnginfo를 추출
    
    Args:
        image: PIL Image 객체
        
    Returns:
        추출된 stealth pnginfo 문자열, 추출 실패 시 None
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