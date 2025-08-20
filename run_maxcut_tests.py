#!/usr/bin/python
# -*-coding:utf-8-*-
import os
import subprocess
import sys

def run_maxcut_test(graph_file, algorithm='goemans_williamson', iterations=50):
    """������ Max Cut ����"""
    cmd = [
        sys.executable, 'maxcut_main.py',
        '--graph_file', graph_file,
        '--algorithm', algorithm,
        '--iterations', str(iterations),
        '--disable_module_output'
    ]
    
    print(f"���չ���: {graph_file}")
    print(f"��k: {algorithm}")
    print(f"���N����: {iterations}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("���զ��\����")
            # �������䵲�G
            lines = result.stdout.split('\n')
            for line in lines:
                if '�̨Τ��έ�' in line or '�`�����ɶ�' in line:
                    print(line)
        else:
            print("���ե���:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("���նW�� (5����)")
    except Exception as e:
        print(f"���եX��: {e}")
    
    print("=" * 60)

def main():
    """����Ҧ�����"""
    graph_dir = "test_graphs"
    
    if not os.path.exists(graph_dir):
        print("�䤣����չϥؿ��A�Х����� create_sample_graphs.py")
        return
    
    # ���o�Ҧ����ɮ�
    graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.txt')]
    
    print(f"��� {len(graph_files)} �Ӵ��չ��ɮ�")
    print("�}�l�������...")
    print("=" * 60)
    
    for graph_file in sorted(graph_files):
        graph_path = os.path.join(graph_dir, graph_file)
        
        # �ھڹϤj�p�վ㭡�N����
        if '200nodes' in graph_file:
            iterations = 20  # �j�ϥθ��֭��N
        elif 'complete' in graph_file:
            iterations = 30  # �����Ϥ������N
        else:
            iterations = 50  # �@��ϸ��h���N
        
        run_maxcut_test(graph_path, 'goemans_williamson', iterations)

if __name__ == '__main__':
    main()
