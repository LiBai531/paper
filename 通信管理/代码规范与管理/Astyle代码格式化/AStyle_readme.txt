�����source insight��ʹ��astyle�Ĵ��������ܣ� 

��source insight �༭����ʱ������source insightû�м��ɵĴ����ʽ�����ߣ� 
GNU��astyle��һ����ѵĴ����ʽ������,�ܹ��������c/c++�淶 ��      
���ǿ��Խ�astyle.exe��ҵ�SourceInsight�С���ϸ�������£� 

1���� http://astyle.sourceforge.net ������AStyle_1.24_windows.zip(����Դ�룬������ϸ�Ķ�)�� 
2����ѹ����bin�ļ����µ�astyle.exe�ŵ�C:\Program Files\astyle�� ���Լ����Է�������λ�ã� 
      ��SourceInsight�˵����Options-->Custom Commands������ѡ��Add,�ڵ����Ի���д�� Astyle, 
3����run�����"C:\Program Files\astyle\astyle.exe" --style=ansi %f��
���У����astyle.exe����·�����пո񣬱����ã���������������--style=ansi ����ansi C ��ʽ
(�������Ҫ��ʽ��java���룬����ط���Ϊ��--style=java),
"%f"��ָ�����ڵ�ǰ�ļ����������ģ���������������Բο�astyle�İ������� 
���Բ鿴��ҳhttp://astyle.sourceforge.net/astyle.html 
4�����⣬�ڴ˽����ϻ�����Ϊ����������ÿ�ݼ��������keys�����������÷���İ�ť�� 
5����SourceInsight�˵����Options-->Menu Assignments�����ϣ�
�������������ΪAstyle��ӵ�ĳ�������£�������Ȼ������Option�£�
�������Command�б����ҵ����Ǹղ���ӵģ�Astyle�����������Menu��ѡ����Ҫ�ӵ��Ǹ��˵��£�
�����Ҽӵ���Option���£������ڣ�Menu Contents��ѡ���ʵ�λ�ã������insert������; 

C:\astyle\bin\Astyle.exe --style=ansi -s8 -S -N -L -m0 -M40 --convert-tabs --suffix=.pre %f

����: http://hi.baidu.com/tihu1111/blog/item/91f77401c9e47a027aec2c63.html 













http://hi.baidu.com/jin_haiqing/blog/item/a3ee7cadb4f08cc27dd92a73.html

ʹ��astyle��ʽ������2010-09-04 21:29astyle��һ������ʽ�����ߣ��������ص�ַ�ǣ�http://sourceforge.net/projects/astyle

һ����������

astyle --style=ansi main.cs (ʹ��ansi����ʽ��main.cs)

�˽����������Ϳ��Ը�ʽ��һ���ļ��ˣ�����������θ�ʽ��Ŀ¼�µ��ļ�

������ʽ��Ŀ¼

for /R %f in (*.cpp;*.cs;) do astyle --style=ansi "%f" (ʹ��ansi����ʽ�µ�ǰĿ¼�µ�����cpp,cs�ļ�,ע�⣺�������ļ�ʱ��"%f" Ҫ��Ϊ"%%f")
for /R %f in (*.c;*.h;) do "D:\Program Files\astyle\bin\astyle" --style=ansi "%f"
��������˵����

(1) -f
�����в���صĴ���֮�������У���import��public class֮�䡢public class�ͳ�Ա֮��ȣ�
(2) -p
�ڲ��������߲���ո���=��+��-�ȡ�
�磺int a=10*60;
�������int a = 10 * 60;
(3) -P
���������߲���ո���-dֻ�������������ո�-Dֻ��������롣
�磺System.out.println(1);
�������System.out.println( 1 );
(4) -U
�Ƴ��������߲���Ҫ�Ŀո�
�磺System.out.println( 1 );
�������System.out.println(1);
(5) -V
��Tab�滻Ϊ�ո�

(6)-N

������Ҫ���namespaces�����û�д˲�����Ч�����£�

namespace foospace
{
class Foo
{
    public:
        Foo();
        virtual ~Foo();
};
}
�д˲����ͻ���������

namespace foospace
{
    class Foo
    {
        public:
            Foo();
            virtual ~Foo();
    };
}

(7) -n

�����ɱ����ļ�����Ĭ�ϵ� .orig�ļ���

C#��Ĭ�Ϸ�ʽΪ�ڶ��֣������������������ʽ��C#����Ļ�����������������ˡ�

�ģ����뵽VS2008,VS2005��

���Ƽ��뵽VS2005��Ҳ��һ��������������û��VS2005,��˵һ��VS2008��������

���ߡ���>�ⲿ���ߡ���>���

���⣺astyle 

���AStyle.exe �����astyle.exe��·����

������--style=allman -N $(ItemDir)$(ItemFileName)$(ItemExt)

��ʼĿ¼��$(TargetDir)

���ϡ�ʹ�ó�ʼĿ¼��

���ȷ����ɡ��Ժ�Ϳ����ڹ��߲˵����ҵ���astyle����һ���ˣ���������Ϳ��ԶԵ�ǰ�ļ����и�ʽ��������


�壺���뵽VS6��

Tools����>Customize����>Tools

���⣺astyle 

���AStyle.exe �����astyle.exe��·����

������--style=ansi -s4 --suffix=.orig $(FileName)$(FileExt)

��ʼĿ¼��$(FileDir)

���ϡ�Using Output Window��

���ȷ����ɡ��Ժ�Ϳ����ڹ��߲˵����ҵ���astyle����һ���ˣ���������Ϳ��ԶԵ�ǰ�ļ����и�ʽ��������


�������뵽Ultraedit��UltraStudio


�߼�-->�������á���>�ⲿ���ߡ���>���

���AStyle.exe -v --style=ansi -s4 --suffix=.orig "%f"�����astyle.exe��·����


Optiones��ѡ�� Windows program��Save Active File.

Output: ѡ��output to list box,show dos box ��no replace��


���ȷ����ɡ��Ժ�Ϳ����ڹ��߲˵����ҵ���astyle����һ���ˣ���������Ϳ��ԶԵ�ǰ�ļ����и�ʽ��������


�ߣ����뵽Source insight


Options-->Custom Command-->Add

Command��astyle

Run "astyle.exe" --style=ansi --pad=oper --unpad=paren -s4 --suffix=.orig %f�����astyle.exe��·����


Output����ѡ.

Control: ѡ��pause when done��exit to window.

source links in output:file, then line

-->menu

add to work menu.


���ȷ����ɡ��Ժ�Ϳ�����Work�˵����ҵ���astyle����һ���ˣ���������Ϳ��ԶԵ�ǰ�ļ����и�ʽ��������

�ˣ�����̨Ŀ¼������(Astyle.bat)


REM bygreencn@gmail.com

REM ��������Ŀ¼�е�����C++�ļ���Astyle���д�����������

REM 2009-01-05

REM ����Astyle����λ�úͲ���

@echo off

set astyle="astyle.exe"

REM ѭ������Ŀ¼

for /r . %%a in (*.cpp;*.c) do %astyle% --style=ansi --pad=oper --unpad=paren -s4 -n "%%a"

for /r . %%a in (*.hpp;*.h) do %astyle% --style=ansi --pad=oper --unpad=paren -s4 -n "%%a"

REM ɾ�����е�astyle�����ļ�

for /r . %%a in (*.orig) do del "%%a"

pause
 




