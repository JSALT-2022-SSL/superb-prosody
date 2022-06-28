import os
from typing import List


class REAPERExtractor(object):
    def __init__(self):
        pass

    def exec(self, wav_path, output_path, fp=0.01):
        """
        Call reaper command, generate 2 files: output_path.f0, ouptut_path.pm.
        """
        os.system(f"reaper -i {wav_path} -f {output_path}.f0 -p {output_path}.pm -a -e {str(fp)}")

    def parse_f0_file(self, path) -> List[float]:
        """
        Parse .f0 file generated from exec() into list of pitch values.
        """
        res = []
        with open(path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line == '\n' or i < 7:
                    continue
                [time, is_defined, pitch] = line.strip().split(' ')
                if is_defined == "0":
                    res.append(0.0)
                else:
                    res.append(float(pitch))
        return res


if __name__ == "__main__":
    extractor = REAPERExtractor()
    extractor.exec(
        wav_path="/mnt/d/Data/LibriTTS/train-clean-100/19/198/19_198_000000_000000.wav",
        output_path="./reaper_test",
        fp=0.01
    )
    input()
    res = extractor.parse_f0_file("./reaper_test.f0")
    print(res)
