# OpenCVApi

In questo progetto viene sviluppata un’Api utilizzando il framework Django in una macchina virtuale con sistema operativo Debian. L’Api verrà utilizzata nel progetto iniziale del sito web e servirà per eseguire un login in modo differente dal consueto. Il login avviene attraverso il riconoscimento facciale, infatti il client, ovvero il sito web, invia all’Api due immagini, una che è salvata nel database e l’atra che viene eseguita nel momento del login. L’Api dovrà confrontare i volti delle due immagini e riconoscere se sono la stessa persona. Lo scopo principale di questo progetto è quello di simulare completamente il funzionamento di un’Api. La macchina virtuale in cui viene eseguita permette di simulare nel client un’Api che potrebbe essere scritta da un altro sviluppatore in una macchina lontana dal client. É stato scelto di utilizzare Djang per sfruttare un’altra caratteristica di un’Api ossia mettere in comunicazione, attraverso il protocollo HTTP, due servizi che utilizzano framework e linguaggi di programmazione diversi. Per lo stesso motivo è stato scelto di eseguire l’Api in un sistema operativo diverso da quello di esecuzione del client, è stato scelto il sistema operativo linux ossia Debian.

Connessione

Il primo ostacolo, il più importante, da superare è relativo alla connessione tra i due servizi sviluppati. Per la corretta connessione tra i due servizi è necessario conoscere l’ip del sistema linux. Successivamente è necessario verificare che almeno una porta del firewall di debian sia aperta per inviare e ricevere dati da un altro sistema, solitamente è la porta 8000. Bisogna poi verificare che i due sistemi siano in comunicazione correttamente, lo si svolge con un ping ossia l’invio e la ricezione di pacchetti di dati tra i due sistemi. Ora è possibile creare l’Uri di partenza che potrebbe essere ad esempio “http://192.168.181.129:8000/”. Esso è formato dall’ip della macchina virutale e dalla porta che è stata aperta e selezionata dal servizio.
Confronto visi
Per eseguire il confronto viene utilizzata la libreria OpenCV. Il sistema riceve in ingresso due immagini con i visi e dovrà verificare se i visi sono della stessa persona restituendo al client lo score della somiglianza. Quando ricevute le immagini, vengono tagliare e modificate in modo da rendere più semplice e corretto il confronto. In particolare viene tagliata l’immagine in modo da avere un’immagine con solo il viso dell’utente e viene convertita in scala di grigi. Successivamente viene calcolato l’indice SSIM tra le due immagini modificate, ossia viene calcato quanto le strutture delle immagini, e quindi dei visi, si assomigliano tra loro. Per calcolare questo indice viene utilizzato il metodo “structural_similarity” della libreria OpenCV.


Metodi

	ConnectionApi: viene utilizzato unicamente per verificare facilmente se il server dell’Api è in funzione e verificare la corretta connessione tra client e server. Accetta richieste “GET” e, se la richiesta è corretta, restituisce lo stato: “Ok”.

	ProcessImage: è un metodo che viene chiamato dal client con una richiesta “POST”. Nel body è richiesto avere due immagini. Sono inizialmente presenti delle verifiche: se la richiesta è corretta, che nel body ci siano effettivamente delle immagini e che sono state correttamente ricevute. Rende le immagini più facilmente confrontabili chiamando il metodo CropFace. Successivamente verrà chiamato il metodo ComparisonFace che restituisce lo score che poi verrà restituito al client come risposta http in formato stringa.

	CropFace: Questo metodo viene eseguito per ogni immagine. Utilizzando il classificatore di Haar per la rilevazione dei volti, vengono rilevati i volti all’interno di un’immagine. Successivamente, verifica se è stato trovato un solo volto e ritaglia l’immagine per avere un’immagine con solo il volto. Se vengono riconosciuti più volti o nessuno, restituisce un messaggio di errore.

	ComparisonFace: Vengono convertite le immagini in scala di grigi e successivamente viene calcolato il grado SSIM (structural similarity index measure) tra le due immagini con l’utilizzo del metodo structural_similarity della libreria OpenCV.


Problematiche

Problemi: Lo score non è molto preciso. In particolare se le immagini inviate ha un’illuminazione o orientazione molto diverse allora molto probabilmente lo score sarà basso e probabilmente il confronto dei visi darà esito negativo.

Soluzioni: Una possibile soluzione potrebbe essere quella di creare un dataset salvato nel database con un elevato numero di immagini che saranno poi confrontate con l’immagine inviata durante la fase di login. Se il dataset è composto da un numero elevato di immagini, lo score sarà più corretto.  Un daraset del genere potrebbe essere popolato anche con le stesse immagini leggermente modificate o ruotate. Un’altra possibile soluzione potrebbe essere quella di creare una AI addestrata a riconoscere quel viso, creando una rete neurale di tipo CNN per ogni utente. L’utilizzo del deep learning è sicuramente la soluzione migliore e più utilizzata in questo ambito ma richiede buone conoscenze di AI. Entrambe le soluzioni richiedono una potenza computazionale molto elevata, ma nella prima soluzione richiede anche una memoria elevata per poter salvare tutto il dataset di immagini per ogni utente.
