# UDP v1


Port domyślny: 12345. Nagranie jest wysyłane **po zakończeniu**, porcjami do 1024 bajtów. ESP32 czeka na ACK każdej porcji; ponawia po 100 ms, maksymalnie 10 prób. Cała wysyłka ma limit 30 s. Odbiornik porządkuje fragmenty, nie transkrybuje niekompletnych nagrań i przez 120 s pamięta zakończone sesje, aby utracony końcowy ACK nie powielał STT.

Wszystkie liczby nagłówka są w network byte order; próbki pozostają PCM16LE.

| Pole | Bajty |
| --- | --- |
| Magic `OLX1` | 4 |
| Wspólny token | 16 |
| Losowy identyfikator sesji | 4 |
| Całkowita długość PCM | 4 |
| Numer porcji od zera | 4 |
| Długość danych tej porcji | 2 |
| PCM | 1–1024 |

ACK: `OLA1` (4) + token (16) + sesja (4) + numer porcji (4). Końcowy ACK oznacza przyjęcie do kolejki odbiornika, nie ukończenie STT. Kolejka jest ograniczona do 4 nagrań, a składanie do 4 sesji. Niepełne sesje wygasają po 5 s bez pakietów lub po 35 s od utworzenia. Duplikaty nie tworzą kolejnych zadań STT w obrębie cache bieżącego procesu. Restart odbiornika nie zachowuje kolejki/cache.

## Adres odbiornika

`RECEIVER_IP` w firmware to **lokalny adres IPv4** odbiornika. Przy uruchomieniu w k3s może to być adres LAN dowolnego węzła — usługa `LoadBalancer` typu UDP jest publikowana przez k3s ServiceLB jako `hostPort` na każdym węźle. Adresy Tailscale widoczne w `status.loadBalancer.ingress` nie są potrzebne ESP32; Arduino nie musi mieć Tailscale. Szczegóły: [`docs/KUBERNETES.md`](KUBERNETES.md).

Uwaga: ServiceLB wykonuje SNAT ruchu LAN, więc opcja `--device-ip` nie działa w tym trybie. Token pozostaje jedynym filtrem nadawcy.

Transport jest przeznaczony do zaufanej sieci LAN: token filtruje przypadkowych nadawców, ale jest przesyłany jawnie i nie zapewnia szyfrowania ani ochrony przed podsłuchem/replay. Nie wystawiaj tego portu do internetu.
