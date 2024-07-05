<h1 align="center">🎯Reconhecimento de Imagens</h1>

<h2> <font size=4.5> O projeto utiliza uma rede neural convolucional treinada para identificar células em imagens obtidas por um microscópio confocal. A iniciativa surgiu da necessidade de analisar imagens individuais de células que, na maioria das vezes, não estão isoladas. </h2>



<p>
    O banco de imagens utilizado no treinamento da rede inclui tanto imagens autorais quanto imagens retiradas do banco de imagens 
    <a href="http://www.cellimagelibrary.org/images/CCDB_6843">Cell Image Library</a>. A imagem abaixo mostra um exemplo de como as imagens estão dispostas.
</p>


<div  align="center">
    <img  src= 'readme_img\ori_and_mask.png' width="400" alt="Descrição da imagem"> </img>
</div>


Para garantir que a máscara e a imagem original tenham o mesmo nome e resolução, é essencial padronizá-las. Optamos pelo tamanho de 1024x1024, já que, em futuros trabalhos, as imagens geralmente serão obtidas nessas dimensões.


<div  align="center">
    <img  src= 'readme_img\code1.png' width="500" alt="Descrição da imagem"> </img>
</div>