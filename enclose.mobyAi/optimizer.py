"""
Post-processing optimizer for GA solutions.
Finds and relocates redundant walls to maximize area.

Mantık:
- Bir duvar "gereksiz" ise, onu kaldırdığımızda Moby hala kaçamıyor demektir
- Bu duvarı başka yere taşıyarak alanı büyütebiliriz
- Buoy budget aynı kalır, verimlilik artar
"""

from typing import List, Tuple, Set, Optional
import numpy as np

from constants import LAND, BUOY, WATER
from analysis import analyze_escape_walls

Coord = Tuple[int, int]


def find_redundant_walls(
    walls: List[Coord],
    grid: np.ndarray,
    moby_pos: Coord
) -> List[Coord]:
    """
    Kaldırılınca hala trapped kalan duvarları bulur.
    
    Bir duvar "gereksiz" ise, onu kaldırdığımızda
    Moby hala kaçamıyor demektir.
    
    Args:
        walls: Mevcut duvar koordinatları
        grid: Harita matrisi
        moby_pos: Moby'nin konumu (x, y)
    
    Returns:
        Gereksiz duvarların listesi
    """
    redundant = []
    walls_set = set(walls)
    
    for wall in walls:
        # Bu duvar olmadan test et
        test_set = walls_set - {wall}
        analysis = analyze_escape_walls(grid, moby_pos, test_set)
        
        if not analysis.escaped:
            # Moby hala hapiste! Bu duvar gereksiz.
            redundant.append(wall)
            
    return redundant


def find_best_relocation(
    wall_to_move: Coord,
    other_walls: List[Coord],
    grid: np.ndarray,
    moby_pos: Coord,
    water_cells: List[Coord]
) -> Tuple[Coord, int]:
    """
    Bir duvarı taşımak için en iyi konumu bulur.
    
    Tüm boş su hücrelerini deneyerek en çok alan
    kazandıran konumu seçer.
    
    Args:
        wall_to_move: Taşınacak duvar
        other_walls: Diğer duvarlar (sabit kalacaklar)
        grid: Harita matrisi
        moby_pos: Moby'nin konumu
        water_cells: Tüm su hücreleri
    
    Returns:
        (en_iyi_konum, en_iyi_alan)
    """
    other_set = set(other_walls)
    best_pos = wall_to_move
    best_area = 0
    
    # Mevcut durumu hesapla (baseline)
    current_set = other_set | {wall_to_move}
    current_analysis = analyze_escape_walls(grid, moby_pos, current_set)
    if not current_analysis.escaped:
        best_area = current_analysis.area
    
    # Tüm boş su hücrelerini dene
    for candidate in water_cells:
        # Zaten duvar var mı? Moby'nin yeri mi?
        if candidate in other_set or candidate == moby_pos:
            continue
        
        # Aynı konum mu?
        if candidate == wall_to_move:
            continue
            
        # Bu konumu test et
        test_set = other_set | {candidate}
        analysis = analyze_escape_walls(grid, moby_pos, test_set)
        
        # Trapped ve daha iyi alan mı?
        if not analysis.escaped and analysis.area > best_area:
            best_area = analysis.area
            best_pos = candidate
            
    return best_pos, best_area


def try_shift_walls(
    walls: List[Coord],
    grid: np.ndarray,
    moby_pos: Coord,
    water_cells_set: Set[Coord]
) -> Tuple[List[Coord], int, bool]:
    """
    Her duvarı 4 yöne kaydırmayı dener.
    
    Bu, GA'nın kaçırdığı küçük iyileştirmeleri yakalar.
    Örnek: Duvarı 1 kare yukarı kaydırınca alan 3 büyüyebilir.
    
    Returns:
        (yeni_duvarlar, yeni_alan, iyileşme_oldu_mu)
    """
    walls_set = set(walls)
    
    # Mevcut alan
    analysis = analyze_escape_walls(grid, moby_pos, walls_set)
    if analysis.escaped:
        return walls, 0, False
    
    current_area = analysis.area
    best_walls = list(walls)
    best_area = current_area
    improved = False
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for i, (wx, wy) in enumerate(walls):
        for dx, dy in directions:
            new_pos = (wx + dx, wy + dy)
            
            # Geçerli konum mu?
            if new_pos not in water_cells_set:
                continue
            if new_pos == moby_pos:
                continue
            if new_pos in walls_set and new_pos != (wx, wy):
                continue
                
            # Test et
            test_walls = walls[:i] + [new_pos] + walls[i+1:]
            test_set = set(test_walls)
            
            analysis = analyze_escape_walls(grid, moby_pos, test_set)
            
            if not analysis.escaped and analysis.area > best_area:
                best_walls = test_walls
                best_area = analysis.area
                improved = True
    
    return best_walls, best_area, improved


def try_multi_swap(
    walls: List[Coord],
    grid: np.ndarray,
    moby_pos: Coord,
    water_cells: List[Coord],
    swap_count: int = 2
) -> Tuple[List[Coord], int, bool]:
    """
    Birden fazla duvarı aynı anda değiştirmeyi dener.

    Bu, single-swap'ın yakalayamadığı iyileştirmeleri bulabilir.
    Örnek: 2 duvarı kaldır, 2 farklı yere koy → daha iyi sonuç

    Args:
        walls: Mevcut duvarlar
        grid: Harita
        moby_pos: Moby konumu
        water_cells: Su hücreleri
        swap_count: Kaç duvar değiştirilecek (2 veya 3)

    Returns:
        (yeni_duvarlar, yeni_alan, iyileşme_oldu_mu)
    """
    import itertools
    import random

    walls_set = set(walls)
    water_set = set(water_cells)

    # Mevcut alan
    analysis = analyze_escape_walls(grid, moby_pos, walls_set)
    if analysis.escaped:
        return walls, 0, False

    current_area = analysis.area
    best_walls = list(walls)
    best_area = current_area
    improved = False

    # Tüm swap_count'lu kombinasyonları dene (çok fazlaysa örnekle)
    wall_indices = list(range(len(walls)))
    all_combos = list(itertools.combinations(wall_indices, swap_count))

    # Çok fazla kombinasyon varsa rastgele örnekle
    if len(all_combos) > 100:
        all_combos = random.sample(all_combos, 100)

    # Potansiyel yeni konumlar (mevcut duvarlar hariç)
    available = [c for c in water_cells if c not in walls_set and c != moby_pos]

    for combo in all_combos:
        # Bu duvarları çıkar
        remaining = [walls[i] for i in range(len(walls)) if i not in combo]
        remaining_set = set(remaining)

        # Yeni konumlar için kombinasyonlar (çok fazlaysa örnekle)
        new_combos = list(itertools.combinations(available, swap_count))
        if len(new_combos) > 200:
            new_combos = random.sample(new_combos, 200)

        for new_positions in new_combos:
            # Yeni duvar seti
            test_walls = remaining + list(new_positions)
            test_set = set(test_walls)

            # Değerlendir
            analysis = analyze_escape_walls(grid, moby_pos, test_set)

            if not analysis.escaped and analysis.area > best_area:
                best_walls = test_walls
                best_area = analysis.area
                improved = True

    return best_walls, best_area, improved


def optimize_solution(
    walls: List[Coord],
    grid: np.ndarray,
    moby_pos: Coord,
    water_cells: List[Coord],
    max_iterations: int = 20,
    verbose: bool = True
) -> Tuple[List[Coord], int]:
    """
    GA çözümünü iteratif olarak iyileştirir.

    Algoritma:
    1. Gereksiz duvarları bul
    2. Her birini en iyi konuma taşı
    3. Shift optimizasyonu uygula
    4. İyileşme olduğu sürece tekrarla

    Args:
        walls: GA'nın bulduğu duvarlar
        grid: Harita matrisi
        moby_pos: Moby'nin konumu
        water_cells: Tüm su hücreleri listesi
        max_iterations: Maksimum iterasyon sayısı
        verbose: Çıktı yazdır

    Returns:
        (optimize_edilmiş_duvarlar, final_alan)
    """
    current_walls = list(walls)
    water_cells_set = set(water_cells)

    # Başlangıç skoru
    current_set = set(current_walls)
    analysis = analyze_escape_walls(grid, moby_pos, current_set)

    if analysis.escaped:
        if verbose:
            print("⚠️ Başlangıç çözümü trapped değil, optimize edilemez.")
        return walls, 0

    current_area = analysis.area

    if verbose:
        print(f"\n{'='*50}")
        print(f"🔧 OPTIMIZER BAŞLADI")
        print(f"{'='*50}")
        print(f"   Başlangıç alanı: {current_area}")
        print(f"   Duvar sayısı: {len(current_walls)}")

    for iteration in range(max_iterations):
        improved_this_round = False

        # === AŞAMA 1: Gereksiz Duvar Tespiti ve Taşıma ===
        redundant = find_redundant_walls(current_walls, grid, moby_pos)

        if verbose and redundant:
            print(f"\n   📍 İterasyon {iteration + 1}: {len(redundant)} gereksiz duvar")

        for wall in redundant:
            # Bu duvarı çıkar
            other_walls = [w for w in current_walls if w != wall]

            # En iyi yeni konumu bul
            new_pos, new_area = find_best_relocation(
                wall, other_walls, grid, moby_pos, water_cells
            )

            if new_area > current_area:
                # İyileşme var!
                current_walls = other_walls + [new_pos]

                if verbose:
                    print(f"      ✨ Taşındı: {wall} → {new_pos}")
                    print(f"         Alan: {current_area} → {new_area} (+{new_area - current_area})")

                current_area = new_area
                improved_this_round = True
                break  # Baştan başla

        # === AŞAMA 2: Shift Optimizasyonu ===
        if not improved_this_round:
            shifted_walls, shifted_area, shift_improved = try_shift_walls(
                current_walls, grid, moby_pos, water_cells_set
            )

            if shift_improved and shifted_area > current_area:
                if verbose:
                    print(f"\n      🔀 Shift iyileştirmesi!")
                    print(f"         Alan: {current_area} → {shifted_area} (+{shifted_area - current_area})")

                current_walls = shifted_walls
                current_area = shifted_area
                improved_this_round = True

        # === AŞAMA 3: Multi-Swap (2 duvar) ===
        if not improved_this_round:
            multi_walls, multi_area, multi_improved = try_multi_swap(
                current_walls, grid, moby_pos, water_cells, swap_count=2
            )

            if multi_improved and multi_area > current_area:
                if verbose:
                    print(f"\n      🔄 Multi-swap (2) iyileştirmesi!")
                    print(f"         Alan: {current_area} → {multi_area} (+{multi_area - current_area})")

                current_walls = multi_walls
                current_area = multi_area
                improved_this_round = True

        # === AŞAMA 4: Multi-Swap (3 duvar) - Sadece takılınca ===
        if not improved_this_round and iteration >= 3:
            multi_walls, multi_area, multi_improved = try_multi_swap(
                current_walls, grid, moby_pos, water_cells, swap_count=3
            )

            if multi_improved and multi_area > current_area:
                if verbose:
                    print(f"\n      🔄 Multi-swap (3) iyileştirmesi!")
                    print(f"         Alan: {current_area} → {multi_area} (+{multi_area - current_area})")

                current_walls = multi_walls
                current_area = multi_area
                improved_this_round = True

        if not improved_this_round:
            if verbose:
                print(f"\n   ✅ Daha fazla iyileştirme bulunamadı.")
            break

    if verbose:
        print(f"\n{'='*50}")
        print(f"🎯 OPTIMIZER SONUÇ")
        print(f"{'='*50}")
        print(f"   Final alan: {current_area}")
        print(f"   Duvarlar: {current_walls}")

    return current_walls, current_area